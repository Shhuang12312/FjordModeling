# =====================================================================================
# fjord.jl — idealized Greenland-fjord large-eddy simulation (Oceananigans)
#
# A rotating, nonhydrostatic LES of a glacial fjord forced by a subglacial-discharge
# plume outflow at the head, with an optional barotropic tide at the mouth. It is a
# stripped-down, general version of the Saqqarleq configuration: every site-specific
# number now lives in one `FjordConfig` block that you edit or override from the command
# line, and the bathymetry is read from a NetCDF file.
#
# The domain is closed (solid walls plus an immersed bottom). The discharge enters as an
# interior relaxation source near the glacier face (y = 0) and leaves through a sponge at
# the mouth (y = Ly), so the net volume flux is zero, which is what the conjugate-gradient
# pressure solver needs.
#
# Four scenarios select the internal-tide forcing:
#   :control    steady discharge, no tide
#   :tide       steady discharge + barotropic M2 tide at the mouth
#   :pump       tidally pulsed discharge, no tide
#   :tidepump   pulsed discharge + tide
#
# Run examples:
#   julia --project fjord.jl --scenario=control --stop_days=7
#   julia --project fjord.jl --scenario=tidepump --architecture=gpu --stop_days=7
#   julia --project fjord.jl --scenario=control --architecture=cpu --Nx=64 --Ny=80 --Nz=40 --stop_days=0.05
# =====================================================================================

using Oceananigans
using Oceananigans.Units
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans.Architectures: on_architecture
using NCDatasets
using Interpolations: linear_interpolation, Flat
using CUDA: has_cuda_gpu
using Printf
using Statistics: mean

# -------------------------------------------------------------------------------------
# 1. Configuration — every knob in one place. Defaults describe a generic Greenland fjord.
# -------------------------------------------------------------------------------------
Base.@kwdef struct FjordConfig
    # run
    name::String         = "control"
    scenario::Symbol     = :control      # :control | :tide | :pump | :tidepump
    architecture::Symbol = :auto         # :cpu | :gpu | :auto
    stop_days::Float64   = 7.0
    output_dir::String   = "output"

    # domain and grid (glacier at y = 0, mouth at y = Ly)
    Lx::Float64 = 12_000.0               # across-fjord width  [m]
    Ly::Float64 = 15_000.0               # along-fjord length  [m]
    Lz::Float64 = 200.0                  # depth               [m]
    Nx::Int = 200
    Ny::Int = 240
    Nz::Int = 100

    # bathymetry: a NetCDF file holding a 2-D bottom-depth array (negative down or
    # positive depth, matching your grid convention). Resampled if it does not match (Nx, Ny).
    bathymetry_file::String = "bottom.nc"
    bathymetry_var::String  = "bottom"

    # physics
    coriolis_f::Float64         = 1.22e-4    # f-plane Coriolis parameter [1/s]
    thermal_expansion::Float64  = 3.87e-5    # alpha [1/degC]
    haline_contraction::Float64 = 7.86e-4    # beta  [1/(g/kg)]

    # ambient stratification: a fresh/warm surface layer over cool/salty deep water,
    # blended with a tanh. Replace T_surface..pycnocline_scale with a fit to a real cast.
    T_surface::Float64        = 3.0          # [degC]
    T_deep::Float64           = 1.0
    S_surface::Float64        = 16.0         # [g/kg]
    S_deep::Float64           = 34.0
    pycnocline_depth::Float64 = 15.0         # centre of the surface-to-deep transition [m]
    pycnocline_scale::Float64 = 8.0          # thickness of the transition [m]

    # subglacial-discharge plume, imposed as an interior source near the glacier face.
    # The source water is given the ambient density of `outflow_depth`, so it is denser
    # than the fresh surface layer and settles there, radiating the internal tide.
    plume_x_center::Float64 = 6_000.0        # across-fjord position of the plume [m]
    plume_x_width::Float64  = 200.0          # across-fjord width of the source   [m]
    plume_y_width::Float64  = 150.0          # along-fjord extent of the source band [m]
    plume_speed::Float64    = 0.34           # peak outflow speed [m/s]
    outflow_depth::Float64  = 15.0           # depth the outflow settles to [m, positive]
    source_top::Float64     = 0.0            # top of the injection band    [m]
    source_bottom::Float64  = -17.0          # bottom of the injection band [m]
    relax_time::Float64     = 60.0           # source relaxation timescale  [s]

    # tide and discharge pulsing
    M2_period::Float64      = 44_700.0       # semidiurnal period [s]
    pump_amplitude::Float64 = 0.5            # fractional discharge modulation (:pump/:tidepump)
    tide_amplitude::Float64 = 0.0168         # barotropic tide velocity at the mouth [m/s]

    # mouth sponge — returns the volume flux so the closed domain stays balanced
    mouth_start::Float64      = 14_700.0     # y where the mouth sponge begins [m]
    mouth_outflow::Float64    = 130 / (20 * 3500)   # mean outflow speed at the mouth [m/s]
    sponge_time::Float64      = 8_000.0      # far-field T/S relaxation timescale [s]
    tide_sponge_time::Float64 = 100.0        # tide-forcing relaxation timescale [s]

    # virtual mooring location
    mooring_x::Float64 = 6_000.0
    mooring_y::Float64 = 350.0

    # output cadence [s]
    mooring_interval::Float64    = 432.0
    section_interval::Float64     = 4_320.0
    average_interval::Float64     = 21_600.0
    checkpoint_interval::Float64  = 4_320.0

    # numerics
    cfl::Float64      = 0.5
    max_dt::Float64   = 30.0
    cg_reltol::Float64 = 1e-4
    cg_maxiter::Int   = 50
end

# Build a config from `--field=value` command-line arguments (any field is overridable).
function config_from_args(args)
    kw = Dict{Symbol,Any}()
    for a in args
        startswith(a, "--") || continue
        parts = split(a[3:end], "="; limit = 2)
        length(parts) == 2 || error("bad argument '$a' (use --field=value)")
        sym = Symbol(parts[1])
        sym in fieldnames(FjordConfig) || error("unknown field --$(parts[1])")
        T = fieldtype(FjordConfig, sym)
        kw[sym] = T === String ? String(parts[2]) :
                  T === Symbol ? Symbol(parts[2]) :
                  T === Int    ? parse(Int, parts[2]) :
                                 parse(Float64, parts[2])
    end
    return FjordConfig(; kw...)
end

# -------------------------------------------------------------------------------------
# 2. Ambient profiles and the plume outflow shape
# -------------------------------------------------------------------------------------
# `p` is an isbits NamedTuple of Float64 scalars so these run inside GPU kernels.
@inline surface_fraction(z, p) = 0.5 * (1 + tanh((z + p.pycnocline_depth) / p.pycnocline_scale))
@inline ambient_T(z, p) = p.T_deep + (p.T_surface - p.T_deep) * surface_fraction(z, p)
@inline ambient_S(z, p) = p.S_deep + (p.S_surface - p.S_deep) * surface_fraction(z, p)

# Outflow speed: a tent peaking at -outflow_depth, tapering to zero at the band edges.
@inline function plume_velocity(z, p)
    zc = -p.outflow_depth
    if z > p.source_top || z < p.source_bottom
        return zero(z)
    elseif z >= zc
        return p.plume_speed * (p.source_top - z) / (p.source_top - zc)
    else
        return p.plume_speed * (z - p.source_bottom) / (zc - p.source_bottom)
    end
end

# -------------------------------------------------------------------------------------
# 3. Source and sponge regions
# -------------------------------------------------------------------------------------
@inline in_source(x, y, z, p) =
    (abs(x - p.plume_x_center) <= 0.5 * p.plume_x_width) &&
    (y <= p.plume_y_width) &&
    (z >= p.source_bottom) && (z <= p.source_top)

# Linear ramp from 0 at the mouth start to 1 at the far wall; used to sponge T/S and the tide.
@inline mouth_ramp(y, p) = y >= p.mouth_start ? (y - p.mouth_start) / (p.Ly - p.mouth_start) : zero(y)
@inline mouth_deep(y, z, p) = z < -5   ? mouth_ramp(y, p) : zero(y)
@inline mouth_upper(y, z, p) = (-20 <= z <= 0) ? mouth_ramp(y, p) : zero(y)

@inline function force_v(x, y, z, t, v, p)
    pulse = 1 + p.pump_amplitude * sin(2π * t / p.M2_period)
    src  = in_source(x, y, z, p) ? -(v - plume_velocity(z, p) * pulse) / p.relax_time : zero(v)
    out  = -mouth_upper(y, z, p) / p.sponge_time * (v - p.mouth_outflow)
    tide = -mouth_deep(y, z, p) / p.tide_sponge_time * (v - p.tide_amplitude * sin(2π * t / p.M2_period))
    return src + out + tide
end

@inline function force_T(x, y, z, t, T, p)
    src = in_source(x, y, z, p) ? -(T - p.T_source) / p.relax_time : zero(T)
    return src - mouth_deep(y, z, p) / p.sponge_time * (T - ambient_T(z, p))
end

@inline function force_S(x, y, z, t, S, p)
    src = in_source(x, y, z, p) ? -(S - p.S_source) / p.relax_time : zero(S)
    return src - mouth_deep(y, z, p) / p.sponge_time * (S - ambient_S(z, p))
end

# -------------------------------------------------------------------------------------
# 4. Grid and bathymetry
# -------------------------------------------------------------------------------------
function build_grid(config, arch)
    underlying = RectilinearGrid(arch;
        size = (config.Nx, config.Ny, config.Nz),
        x = (0, config.Lx), y = (0, config.Ly), z = (-config.Lz, 0),
        halo = (4, 4, 4), topology = (Bounded, Bounded, Bounded))

    path = isabspath(config.bathymetry_file) ? config.bathymetry_file :
           joinpath(@__DIR__, config.bathymetry_file)
    B = NCDataset(path) do ds
        Array{Float64}(ds[config.bathymetry_var][:, :])
    end

    if size(B) == (config.Nx, config.Ny)
        bottom = on_architecture(arch, B)
    else
        nbx, nby = size(B)
        itp = linear_interpolation(
            (range(0, config.Lx; length = nbx), range(0, config.Ly; length = nby)),
            B; extrapolation_bc = Flat())
        xc = [(i - 0.5) * config.Lx / config.Nx for i in 1:config.Nx]
        yc = [(j - 0.5) * config.Ly / config.Ny for j in 1:config.Ny]
        bottom = on_architecture(arch, [itp(xc[i], yc[j]) for i in 1:config.Nx, j in 1:config.Ny])
        @warn "bathymetry $(size(B)) resampled onto the ($(config.Nx), $(config.Ny)) grid"
    end

    return ImmersedBoundaryGrid(underlying, GridFittedBottom(bottom))
end

# -------------------------------------------------------------------------------------
# 5. Model
# -------------------------------------------------------------------------------------
function build_model(config, grid)
    pump_on = config.scenario in (:pump, :tidepump)
    tide_on = config.scenario in (:tide, :tidepump)

    # isbits NamedTuple passed to every forcing/kernel (GPU-safe: Float64 only)
    p = (pycnocline_depth = config.pycnocline_depth, pycnocline_scale = config.pycnocline_scale,
         T_surface = config.T_surface, T_deep = config.T_deep,
         S_surface = config.S_surface, S_deep = config.S_deep,
         plume_x_center = config.plume_x_center, plume_x_width = config.plume_x_width,
         plume_y_width = config.plume_y_width, plume_speed = config.plume_speed,
         outflow_depth = config.outflow_depth, source_top = config.source_top,
         source_bottom = config.source_bottom, relax_time = config.relax_time,
         Ly = config.Ly, mouth_start = config.mouth_start, mouth_outflow = config.mouth_outflow,
         sponge_time = config.sponge_time, tide_sponge_time = config.tide_sponge_time,
         M2_period = config.M2_period,
         pump_amplitude = pump_on ? config.pump_amplitude : 0.0,
         tide_amplitude = tide_on ? config.tide_amplitude : 0.0,
         # source T/S set to the ambient density of the settling depth (negatively buoyant)
         T_source = ambient_T(-config.outflow_depth,
                              (; pycnocline_depth = config.pycnocline_depth,
                                 pycnocline_scale = config.pycnocline_scale,
                                 T_surface = config.T_surface, T_deep = config.T_deep)),
         S_source = ambient_S(-config.outflow_depth,
                              (; pycnocline_depth = config.pycnocline_depth,
                                 pycnocline_scale = config.pycnocline_scale,
                                 S_surface = config.S_surface, S_deep = config.S_deep)))

    forcing = (v = Forcing(force_v; parameters = p, field_dependencies = :v),
               T = Forcing(force_T; parameters = p, field_dependencies = :T),
               S = Forcing(force_S; parameters = p, field_dependencies = :S))

    boundary_conditions = (
        u = FieldBoundaryConditions(immersed = ValueBoundaryCondition(0)),
        v = FieldBoundaryConditions(immersed = ValueBoundaryCondition(0)),
        T = FieldBoundaryConditions(top = FluxBoundaryCondition(0)),
        S = FieldBoundaryConditions(top = FluxBoundaryCondition(0)))

    eos = LinearEquationOfState(thermal_expansion = config.thermal_expansion,
                                haline_contraction = config.haline_contraction)

    model = NonhydrostaticModel(grid;
        advection = WENO(order = 5),                 # implicit LES; no extra closure needed
        timestepper = :RungeKutta3,
        tracers = (:T, :S),
        buoyancy = SeawaterBuoyancy(equation_of_state = eos),
        coriolis = FPlane(f = config.coriolis_f),
        hydrostatic_pressure_anomaly = CenterField(grid),
        forcing = forcing,
        boundary_conditions = boundary_conditions,
        pressure_solver = ConjugateGradientPoissonSolver(grid;
            reltol = config.cg_reltol, maxiter = config.cg_maxiter))

    return model, p
end

# -------------------------------------------------------------------------------------
# 6. Initial condition, callbacks, and outputs
# -------------------------------------------------------------------------------------
function set_initial_condition!(model, config, p)
    u, v, w = model.velocities
    noise() = 5e-3 .* (rand(size(u)...) .- 0.5)
    uᵢ, vᵢ, wᵢ = noise(), noise(), noise()
    uᵢ .-= mean(uᵢ); vᵢ .-= mean(vᵢ); wᵢ .-= mean(wᵢ)
    set!(model, u = uᵢ, v = vᵢ, w = wᵢ,
         T = (x, y, z) -> ambient_T(z, p), S = (x, y, z) -> ambient_S(z, p))
end

function log_progress(sim)
    u, v, w = sim.model.velocities
    umax = maximum(abs, interior(u))
    vmax = maximum(abs, interior(v))
    wmax = maximum(abs, interior(w))
    @info @sprintf("t = %6.2f days   Δt = %5.1f s   max|u,v,w| = (%.3f, %.3f, %.3f)",
                   sim.model.clock.time / 86400, sim.Δt, umax, vmax, wmax)
end

nearest_index(target, L, N) = clamp(round(Int, target / L * N), 1, N)

function add_outputs!(simulation, model, config)
    u, v, w = model.velocities
    T, S = model.tracers
    ω_y = Field(∂z(u) - ∂x(w))                # along-fjord vorticity, handy for waves
    outputs = (; u, v, w, T, S, ω_y)

    outdir = config.output_dir
    mkpath(outdir)
    prefix = config.name
    ckpt = "checkpoint_" * prefix
    pickup = any(startswith("$(ckpt)_iteration"), readdir(outdir))
    overwrite = !pickup
    pickup && @warn "checkpoint for $prefix found in $outdir — resuming"

    ix = nearest_index(config.mooring_x, config.Lx, config.Nx)
    iy = nearest_index(config.mooring_y, config.Ly, config.Ny)
    ksurf = config.Nz
    kout  = nearest_index(config.Lz - config.outflow_depth, config.Lz, config.Nz)

    # full-depth column at the mooring
    simulation.output_writers[:mooring] = NetCDFWriter(model, outputs;
        filename = joinpath(outdir, "$(prefix)_mooring.nc"),
        schedule = TimeInterval(config.mooring_interval),
        indices = (ix, iy, :), overwrite_existing = overwrite)

    # along-fjord section through the mooring
    simulation.output_writers[:section] = NetCDFWriter(model, outputs;
        filename = joinpath(outdir, "$(prefix)_section.nc"),
        schedule = TimeInterval(config.section_interval),
        indices = (ix, :, :), overwrite_existing = overwrite)

    # horizontal slices at the surface and at the outflow depth
    simulation.output_writers[:surface] = NetCDFWriter(model, outputs;
        filename = joinpath(outdir, "$(prefix)_surface.nc"),
        schedule = TimeInterval(config.section_interval),
        indices = (:, :, ksurf), overwrite_existing = overwrite)
    simulation.output_writers[:outflow] = NetCDFWriter(model, outputs;
        filename = joinpath(outdir, "$(prefix)_outflow.nc"),
        schedule = TimeInterval(config.section_interval),
        indices = (:, :, kout), overwrite_existing = overwrite)

    # time-averaged fields
    simulation.output_writers[:average] = NetCDFWriter(model, (; u, v, w, T, S);
        filename = joinpath(outdir, "$(prefix)_average.nc"),
        schedule = AveragedTimeInterval(config.average_interval, window = config.average_interval),
        overwrite_existing = overwrite)

    # restart checkpoints
    simulation.output_writers[:checkpointer] = Checkpointer(model;
        schedule = TimeInterval(config.checkpoint_interval),
        dir = outdir, prefix = ckpt, cleanup = true)

    return pickup
end

# -------------------------------------------------------------------------------------
# 7. Run
# -------------------------------------------------------------------------------------
function main(args = ARGS)
    config = config_from_args(args)
    arch = config.architecture == :cpu ? CPU() :
           config.architecture == :gpu ? GPU() :
           (has_cuda_gpu() ? GPU() : CPU())
    @info "Greenland fjord LES" name=config.name scenario=config.scenario architecture=arch

    grid = build_grid(config, arch)
    model, p = build_model(config, grid)
    set_initial_condition!(model, config, p)

    simulation = Simulation(model; Δt = 0.5, stop_time = config.stop_days * days)
    simulation.callbacks[:wizard] = Callback(
        TimeStepWizard(cfl = config.cfl, max_change = 1.05, min_change = 0.2, max_Δt = config.max_dt),
        IterationInterval(2))
    simulation.callbacks[:progress] = Callback(log_progress, IterationInterval(20))

    pickup = add_outputs!(simulation, model, config)

    @info "starting run" stop_days=config.stop_days pickup
    run!(simulation; pickup)
    @info "done: $(config.name)"
end

main()
