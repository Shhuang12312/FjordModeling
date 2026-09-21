# =====================================================================================
# fjord.jl - idealized Greenland-fjord large-eddy simulation (LES) with Oceananigans
#
# WHAT THIS MODEL IS
# ------------------
# This is a fluid-dynamics simulation of a glacial fjord: a long, deep, narrow ocean
# inlet with a glacier at its head. Meltwater collects at the base of the glacier and
# erupts into the fjord as a "subglacial discharge plume." That plume, and the tides,
# stir up the fjord and can launch internal waves (waves that ride along the boundary
# between light surface water and dense deep water, rather than on the sea surface).
# The goal is to see how the circulation and the internal tide respond to the plume.
#
# "Large-eddy simulation" means we resolve the big, energy-containing turbulent eddies
# directly on the grid and let a numerical scheme handle the tiny unresolved ones.
# "Nonhydrostatic" means we solve the full vertical momentum equation (needed because
# plumes and steep internal waves have strong vertical accelerations that the usual
# hydrostatic approximation would miss). We use the Boussinesq approximation: density
# differences matter only for buoyancy, and are treated as small everywhere else.
#
# THE FOUR EXPERIMENTS
# --------------------
# We compare four runs to learn what actually drives the internal tide:
#   :control    steady discharge, no tide                (baseline)
#   :tide       steady discharge + a tide at the mouth   (does the tide alone do it?)
#   :pump       discharge pulsed at the tidal period     (does pulsing the plume do it?)
#   :tidepump   pulsed discharge + tide                  (the realistic case)
# Comparing them isolates the local plume-driven internal tide from the basin-scale tide.
#
# WHERE THE REAL-WORLD DATA GOES (swap these in when you have them)
# ----------------------------------------------------------------
#   Bathymetry (fjord shape):   GEBCO gridded bathymetry -> bottom.nc  (see build_grid
#                               and make_bathymetry.jl). GEBCO stores "elevation" in
#                               metres, negative below sea level, which is exactly the
#                               bottom-height convention the model wants.
#   Ambient temperature/salt:   ASTE state estimate -> set the ambient_* parameters in
#                               FjordConfig by fitting a representative ASTE profile near
#                               the fjord mouth (see the "ambient stratification" block).
#   Subglacial discharge:       RACMO 2.3p2 runoff (Zenodo 19254871) -> sets how strong
#                               the plume outflow is (plume_speed) and, seasonally, which
#                               period you are modelling (see the plume-source block).
#
# HOW TO RUN
# ----------
#   julia --project fjord.jl --scenario=control --stop_days=7
#   julia --project fjord.jl --scenario=tidepump --architecture=gpu --stop_days=7
#   # fast, coarse CPU test to check everything wires together (~1 model-hour):
#   julia --project fjord.jl --scenario=control --architecture=cpu --Nx=64 --Ny=80 --Nz=40 --stop_days=0.05
# =====================================================================================

using Oceananigans                 # the ocean model itself
using Oceananigans.Units           # gives us `days`, so we can write times in human units
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans.Architectures: on_architecture   # moves arrays to the CPU or GPU
using NCDatasets                   # read/write NetCDF (.nc) files, the standard in oceanography
using Interpolations: linear_interpolation, Flat     # regrid bathymetry onto the model grid
using CUDA: has_cuda_gpu           # detect whether a GPU is present
using Printf                       # nicely formatted progress lines
using Statistics: mean

# -------------------------------------------------------------------------------------
# 1. Configuration
# -------------------------------------------------------------------------------------
# Every adjustable number lives here, grouped and labelled with its physical units. To
# model a different fjord or a different season, you change values here (or override any
# of them on the command line as --field=value); you should not need to touch the code
# below. `Base.@kwdef` turns each line into a named default you can override.
Base.@kwdef struct FjordConfig
    # --- run bookkeeping ---
    name::String         = "control"     # label used to name the output files
    scenario::Symbol     = :control      # :control | :tide | :pump | :tidepump (see header)
    architecture::Symbol = :auto         # :cpu | :gpu | :auto (auto = GPU if one exists)
    stop_days::Float64   = 7.0           # how many days of fjord time to simulate
    output_dir::String   = "output"      # folder the .nc files are written to

    # --- domain and grid ---
    # The box is Lx wide (across the fjord), Ly long (glacier at y=0, mouth at y=Ly), and
    # Lz deep. Nx/Ny/Nz set how many grid cells span each direction: more cells = finer
    # resolution but more computation. Default cell size is ~50 m across/along and 2 m
    # vertically, which resolves the plume and the internal tide in a fjord this size.
    Lx::Float64 = 12_000.0               # across-fjord width  [m]
    Ly::Float64 = 15_000.0               # along-fjord length  [m]
    Lz::Float64 = 200.0                  # depth               [m]
    Nx::Int = 200
    Ny::Int = 240
    Nz::Int = 100

    # --- bathymetry (fjord bottom shape) ---
    # A NetCDF file with a 2-D bottom-height array. Use GEBCO: download the fjord's
    # region, and the model reads the depth field named by `bathymetry_var`. The array
    # does not have to match the model grid; build_grid resamples it if needed.
    bathymetry_file::String = "bottom.nc"
    bathymetry_var::String  = "bottom"   # GEBCO's own files usually name this "elevation"

    # --- physics constants ---
    coriolis_f::Float64         = 1.22e-4    # Earth's rotation felt at this latitude [1/s]
    thermal_expansion::Float64  = 3.87e-5    # how much water expands when warmed, alpha [1/degC]
    haline_contraction::Float64 = 7.86e-4    # how much water shrinks when saltier, beta [1/(g/kg)]

    # --- ambient stratification (the background temperature/salinity of the fjord) ---
    # Fjord water is layered: fresh, and in summer warm, near the surface; cold and salty
    # at depth. We describe that with a smooth transition (a tanh) from surface values to
    # deep values. Set these five numbers by fitting a representative ASTE profile taken
    # near the fjord mouth: read T and S versus depth, then choose surface/deep values and
    # the depth and thickness of the layer in between. (An analytic profile like this is
    # used because it must be evaluated cheaply at every grid point, including on the GPU.)
    T_surface::Float64        = 3.0          # near-surface temperature [degC]
    T_deep::Float64           = 1.0          # deep temperature        [degC]
    S_surface::Float64        = 16.0         # near-surface salinity    [g/kg]
    S_deep::Float64           = 34.0         # deep salinity            [g/kg]
    pycnocline_depth::Float64 = 15.0         # depth of the surface-to-deep transition [m]
    pycnocline_scale::Float64 = 8.0          # how thick that transition is            [m]

    # --- subglacial-discharge plume ---
    # The rising plume at the glacier face is handled by a separate, finer plume model; here
    # we inject its *outflow* as a patch of water near the glacier (y near 0). The trick that
    # makes it launch internal waves: we give the injected water the density it would have at
    # `outflow_depth`, so it is heavier than the fresh surface layer, sinks to that depth, and
    # settles there. `plume_speed` is how fast that outflow moves; scale it with the RACMO
    # 2.3p2 subglacial discharge for the season you are modelling (a bigger meltwater flux
    # means a faster, stronger outflow).
    plume_x_center::Float64 = 6_000.0        # where across the fjord the plume enters [m]
    plume_x_width::Float64  = 200.0          # how wide the injected patch is (across) [m]
    plume_y_width::Float64  = 150.0          # how far the patch reaches from the glacier [m]
    plume_speed::Float64    = 0.34           # peak outflow speed [m/s]
    outflow_depth::Float64  = 15.0           # depth the outflow settles to [m below surface]
    source_top::Float64     = 0.0            # top of the injection band    [m]
    source_bottom::Float64  = -17.0          # bottom of the injection band [m]
    relax_time::Float64     = 60.0           # how quickly we nudge the patch to target [s]

    # --- tide and discharge pulsing ---
    # M2 is the main lunar semidiurnal tide (one cycle every 12.42 h). In the :pump runs we
    # multiply the discharge by (1 + pump_amplitude * sin(2*pi*t/M2)) so it strengthens and
    # weakens each tidal cycle. In the :tide runs we push a weak back-and-forth flow at the
    # mouth to represent the basin tide reaching the fjord.
    M2_period::Float64      = 44_700.0       # tidal period [s] (12.42 hours)
    pump_amplitude::Float64 = 0.5            # fractional swing of the discharge (:pump/:tidepump)
    tide_amplitude::Float64 = 0.0168         # tidal current speed at the mouth [m/s]

    # --- mouth sponge ---
    # The domain is closed (walls all around), so whatever volume the plume adds must leave
    # somewhere or the fjord would "fill up." Near the mouth we gently relax the flow toward
    # a slow outflow and relax T/S back to ambient, which lets water exit and keeps the net
    # volume balanced. "Relax" means: add a force that nudges a field toward a target over a
    # chosen timescale (a longer timescale = gentler nudge).
    mouth_start::Float64      = 14_700.0     # where the mouth sponge begins [m]
    mouth_outflow::Float64    = 130 / (20 * 3500)   # target mean outflow at the mouth [m/s]
    sponge_time::Float64      = 8_000.0      # gentle T/S relaxation timescale [s]
    tide_sponge_time::Float64 = 100.0        # faster relaxation used to force the tide [s]

    # --- virtual mooring (a fake instrument we "hang" in the fjord to record time series) ---
    mooring_x::Float64 = 6_000.0
    mooring_y::Float64 = 350.0

    # --- how often to save output, in seconds ---
    mooring_interval::Float64    = 432.0     # the mooring column: often, to resolve the tide
    section_interval::Float64     = 4_320.0  # 2-D slices: less often (they are larger files)
    average_interval::Float64     = 21_600.0 # window for time-averaged fields
    checkpoint_interval::Float64  = 4_320.0  # how often to save a restart checkpoint

    # --- numerics ---
    # The timestep is chosen automatically to keep the simulation stable (the "CFL"
    # condition: a parcel must not cross more than a fraction of a cell per step).
    cfl::Float64       = 0.5                 # target fraction of a cell crossed per step
    max_dt::Float64    = 30.0                # never let the timestep exceed this [s]
    cg_reltol::Float64 = 1e-4                # pressure-solver tolerance (smaller = stricter)
    cg_maxiter::Int    = 50                  # pressure-solver iteration cap per step
end

# Read `--field=value` pairs from the command line and override the matching defaults.
# This lets you launch many runs from a script without editing the file each time.
function config_from_args(args)
    kw = Dict{Symbol,Any}()
    for a in args
        startswith(a, "--") || continue
        parts = split(a[3:end], "="; limit = 2)
        length(parts) == 2 || error("bad argument '$a' (use --field=value)")
        sym = Symbol(parts[1])
        sym in fieldnames(FjordConfig) || error("unknown field --$(parts[1])")
        T = fieldtype(FjordConfig, sym)      # convert the text to the field's type
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
# These small functions are called at every grid point, so they must be plain arithmetic
# (no file reads, no lookups) to run on the GPU. `p` is a lightweight bundle of the
# numbers they need. `@inline` asks the compiler to paste the function body in where it
# is called, which is faster for such tiny functions.

# Fraction of "surface character" at height z: ~1 near the surface, ~0 in the deep water,
# with a smooth tanh transition centred at -pycnocline_depth. (z is height, so it is 0 at
# the surface and negative below.)
@inline surface_fraction(z, p) = 0.5 * (1 + tanh((z + p.pycnocline_depth) / p.pycnocline_scale))
@inline ambient_T(z, p) = p.T_deep + (p.T_surface - p.T_deep) * surface_fraction(z, p)
@inline ambient_S(z, p) = p.S_deep + (p.S_surface - p.S_deep) * surface_fraction(z, p)

# The outflow speed as a function of depth: zero outside the injection band, rising to a
# peak of `plume_speed` at the settling depth (-outflow_depth), and tapering linearly to
# zero at the top and bottom of the band. Picture a tent whose peak sits at the outflow depth.
@inline function plume_velocity(z, p)
    zc = -p.outflow_depth
    if z > p.source_top || z < p.source_bottom
        return zero(z)                                   # outside the band: no injection
    elseif z >= zc
        return p.plume_speed * (p.source_top - z) / (p.source_top - zc)   # peak -> 0 upward
    else
        return p.plume_speed * (z - p.source_bottom) / (zc - p.source_bottom)  # 0 -> peak
    end
end

# -------------------------------------------------------------------------------------
# 3. Source and sponge regions (the "forcing" that drives the fjord)
# -------------------------------------------------------------------------------------
# Oceananigans lets us add a forcing term to each equation: an extra tendency evaluated at
# every point and time. We use it two ways: (1) inject the plume near the glacier, and
# (2) sponge the flow near the mouth so water can leave and T/S stay realistic.

# True inside the small box near the glacier where we inject the plume outflow.
@inline in_source(x, y, z, p) =
    (abs(x - p.plume_x_center) <= 0.5 * p.plume_x_width) &&   # within the across-fjord patch
    (y <= p.plume_y_width) &&                                 # close to the glacier (y ~ 0)
    (z >= p.source_bottom) && (z <= p.source_top)            # within the injection depth band

# A ramp that is 0 at the start of the mouth sponge and grows to 1 at the far wall, so the
# nudging turns on gradually (a sudden wall of forcing would bounce waves back).
@inline mouth_ramp(y, p)     = y >= p.mouth_start ? (y - p.mouth_start) / (p.Ly - p.mouth_start) : zero(y)
@inline mouth_deep(y, z, p)  = z < -5 ? mouth_ramp(y, p) : zero(y)          # below the surface layer
@inline mouth_upper(y, z, p) = (-20 <= z <= 0) ? mouth_ramp(y, p) : zero(y) # the upper outflow layer

# Forcing on the along-fjord velocity v. Three additive pieces:
#   src : inside the source box, nudge v toward the (possibly pulsing) plume outflow speed.
#   out : near the mouth's upper layer, nudge v toward a slow steady outflow (mass balance).
#   tide: near the mouth's deeper layer, nudge v toward a back-and-forth tidal current.
@inline function force_v(x, y, z, t, v, p)
    pulse = 1 + p.pump_amplitude * sin(2π * t / p.M2_period)   # 1.0 in steady runs
    src  = in_source(x, y, z, p) ? -(v - plume_velocity(z, p) * pulse) / p.relax_time : zero(v)
    out  = -mouth_upper(y, z, p) / p.sponge_time * (v - p.mouth_outflow)
    tide = -mouth_deep(y, z, p)  / p.tide_sponge_time * (v - p.tide_amplitude * sin(2π * t / p.M2_period))
    return src + out + tide
end

# Forcing on temperature: inject the plume's temperature in the source box, and relax back
# to the ambient profile near the mouth. p.T_source is the plume water's temperature (set to
# the ambient value at the settling depth so the outflow is neutrally buoyant once it arrives).
@inline function force_T(x, y, z, t, T, p)
    src = in_source(x, y, z, p) ? -(T - p.T_source) / p.relax_time : zero(T)
    return src - mouth_deep(y, z, p) / p.sponge_time * (T - ambient_T(z, p))
end

# Forcing on salinity: the same idea. Salinity is what mainly sets density in these fjords,
# so injecting the right salinity is what makes the outflow sink to `outflow_depth`.
@inline function force_S(x, y, z, t, S, p)
    src = in_source(x, y, z, p) ? -(S - p.S_source) / p.relax_time : zero(S)
    return src - mouth_deep(y, z, p) / p.sponge_time * (S - ambient_S(z, p))
end

# -------------------------------------------------------------------------------------
# 4. Grid and bathymetry
# -------------------------------------------------------------------------------------
# We first build a plain rectangular box, then carve the fjord's real bottom out of it
# using an "immersed boundary": grid cells below the sea floor are switched off so the
# water feels the true shape of the fjord.
function build_grid(config, arch)
    # A rectangular grid of cells. `Bounded` means solid walls on every side (a closed box).
    # `halo = (4, 4, 4)` are extra ghost cells around the edges that the advection scheme needs.
    underlying = RectilinearGrid(arch;
        size = (config.Nx, config.Ny, config.Nz),
        x = (0, config.Lx), y = (0, config.Ly), z = (-config.Lz, 0),
        halo = (4, 4, 4), topology = (Bounded, Bounded, Bounded))

    # Read the bottom-height field from the NetCDF file (GEBCO, once you have it).
    path = isabspath(config.bathymetry_file) ? config.bathymetry_file :
           joinpath(@__DIR__, config.bathymetry_file)
    B = NCDataset(path) do ds
        Array{Float64}(ds[config.bathymetry_var][:, :])
    end

    if size(B) == (config.Nx, config.Ny)
        bottom = on_architecture(arch, B)                # already on our grid: use as-is
    else
        # Otherwise interpolate the file's grid onto our model grid. This is why you can
        # feed in raw GEBCO at its own resolution and still run any Nx, Ny you like.
        nbx, nby = size(B)
        itp = linear_interpolation(
            (range(0, config.Lx; length = nbx), range(0, config.Ly; length = nby)),
            B; extrapolation_bc = Flat())
        xc = [(i - 0.5) * config.Lx / config.Nx for i in 1:config.Nx]   # cell-centre positions
        yc = [(j - 0.5) * config.Ly / config.Ny for j in 1:config.Ny]
        bottom = on_architecture(arch, [itp(xc[i], yc[j]) for i in 1:config.Nx, j in 1:config.Ny])
        @warn "bathymetry $(size(B)) resampled onto the ($(config.Nx), $(config.Ny)) grid"
    end

    # GridFittedBottom switches off the cells that lie below `bottom`, giving the fjord its shape.
    return ImmersedBoundaryGrid(underlying, GridFittedBottom(bottom))
end

# -------------------------------------------------------------------------------------
# 5. Model
# -------------------------------------------------------------------------------------
# Assemble the physics: advection, rotation, buoyancy, the pressure solver, the forcing,
# and the boundary conditions. This is where all the pieces come together into one model.
function build_model(config, grid)
    pump_on = config.scenario in (:pump, :tidepump)   # is the discharge pulsed?
    tide_on = config.scenario in (:tide, :tidepump)   # is the mouth tide on?

    # Bundle the scalars the forcing needs into one lightweight, GPU-friendly package.
    # (It holds only plain numbers, which is what lets it be used inside GPU kernels.)
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
         # Turn pumping/tide off (amplitude 0) unless this scenario calls for them.
         pump_amplitude = pump_on ? config.pump_amplitude : 0.0,
         tide_amplitude = tide_on ? config.tide_amplitude : 0.0,
         # The plume water's T and S: the ambient values at the settling depth, so the
         # injected water is denser than the surface layer, sinks, and is neutrally buoyant
         # once it reaches `outflow_depth`.
         T_source = ambient_T(-config.outflow_depth,
                              (; pycnocline_depth = config.pycnocline_depth,
                                 pycnocline_scale = config.pycnocline_scale,
                                 T_surface = config.T_surface, T_deep = config.T_deep)),
         S_source = ambient_S(-config.outflow_depth,
                              (; pycnocline_depth = config.pycnocline_depth,
                                 pycnocline_scale = config.pycnocline_scale,
                                 S_surface = config.S_surface, S_deep = config.S_deep)))

    # Attach each forcing function to its field. `field_dependencies` tells Oceananigans
    # which field to pass in (so force_v receives the current v, etc.).
    forcing = (v = Forcing(force_v; parameters = p, field_dependencies = :v),
               T = Forcing(force_T; parameters = p, field_dependencies = :T),
               S = Forcing(force_S; parameters = p, field_dependencies = :S))

    # Boundary conditions: no flow through the immersed bottom, and no heat/salt through the
    # sea surface (a flux of 0). The side walls are handled by the `Bounded` topology.
    boundary_conditions = (
        u = FieldBoundaryConditions(immersed = ValueBoundaryCondition(0)),
        v = FieldBoundaryConditions(immersed = ValueBoundaryCondition(0)),
        T = FieldBoundaryConditions(top = FluxBoundaryCondition(0)),
        S = FieldBoundaryConditions(top = FluxBoundaryCondition(0)))

    # A linear equation of state: density depends linearly on temperature and salinity
    # through the two coefficients above. Simple, fast, and good enough for a fjord.
    eos = LinearEquationOfState(thermal_expansion = config.thermal_expansion,
                                haline_contraction = config.haline_contraction)

    # The model. Notes on the less-obvious choices:
    #   WENO(order = 5)  a high-order advection scheme; its built-in numerical dissipation
    #                    plays the role of the small-eddy turbulence closure (implicit LES),
    #                    so we do not add a separate closure.
    #   RungeKutta3      a stable, accurate three-stage time-stepping scheme.
    #   hydrostatic_pressure_anomaly = CenterField(grid)  splits the large, simple hydrostatic
    #                    part of the pressure off analytically so the solver only has to work
    #                    on the small remainder. This is a key stability/speed trick.
    #   ConjugateGradientPoissonSolver  finds the pressure that keeps the flow incompressible.
    #                    Unlike the default Fourier solver, it respects the immersed bottom,
    #                    which removes a spurious tracer signal that otherwise appears over
    #                    the sloping sea floor.
    model = NonhydrostaticModel(grid;
        advection = WENO(order = 5),
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
# 6. Initial condition, progress reporting, and outputs
# -------------------------------------------------------------------------------------
# Start the fjord at rest, filled with the ambient stratification, plus a whisper of random
# velocity. The noise breaks the perfect symmetry so realistic turbulence can develop.
function set_initial_condition!(model, config, p)
    u, v, w = model.velocities
    noise() = 5e-3 .* (rand(size(u)...) .- 0.5)     # tiny random velocities, +/-2.5 mm/s
    uᵢ, vᵢ, wᵢ = noise(), noise(), noise()
    uᵢ .-= mean(uᵢ); vᵢ .-= mean(vᵢ); wᵢ .-= mean(wᵢ)   # remove any net drift
    set!(model, u = uᵢ, v = vᵢ, w = wᵢ,
         T = (x, y, z) -> ambient_T(z, p), S = (x, y, z) -> ambient_S(z, p))
end

# Print a one-line status every so often so you can watch the run and catch blow-ups early.
# A healthy run keeps the timestep and the maximum speeds steady; runaway numbers mean trouble.
function log_progress(sim)
    u, v, w = sim.model.velocities
    umax = maximum(abs, interior(u))
    vmax = maximum(abs, interior(v))
    wmax = maximum(abs, interior(w))
    @info @sprintf("t = %6.2f days   Δt = %5.1f s   max|u,v,w| = (%.3f, %.3f, %.3f) m/s",
                   sim.model.clock.time / 86400, sim.Δt, umax, vmax, wmax)
end

# Convert a physical position (metres) to the nearest grid-cell index, clamped to the grid.
nearest_index(target, L, N) = clamp(round(Int, target / L * N), 1, N)

# Set up the files we save. Each "output writer" records a chosen slice of the fields on its
# own schedule. Saving slices instead of the whole 3-D field every time keeps files manageable.
function add_outputs!(simulation, model, config)
    u, v, w = model.velocities
    T, S = model.tracers
    ω_y = Field(∂z(u) - ∂x(w))                       # along-fjord vorticity: makes waves visible
    outputs = (; u, v, w, T, S, ω_y)

    outdir = config.output_dir
    mkpath(outdir)
    prefix = config.name
    ckpt = "checkpoint_" * prefix
    # If a checkpoint for this run already exists, we will resume from it and append to the
    # existing files; otherwise we start fresh and overwrite. This makes long runs restartable.
    pickup = any(startswith("$(ckpt)_iteration"), readdir(outdir))
    overwrite = !pickup
    pickup && @warn "checkpoint for $prefix found in $outdir - resuming"

    ix = nearest_index(config.mooring_x, config.Lx, config.Nx)     # mooring column indices
    iy = nearest_index(config.mooring_y, config.Ly, config.Ny)
    ksurf = config.Nz                                              # top (surface) cell
    kout  = nearest_index(config.Lz - config.outflow_depth, config.Lz, config.Nz)  # outflow depth

    # A full-depth column at the mooring, saved often to resolve the tide there.
    simulation.output_writers[:mooring] = NetCDFWriter(model, outputs;
        filename = joinpath(outdir, "$(prefix)_mooring.nc"),
        schedule = TimeInterval(config.mooring_interval),
        indices = (ix, iy, :), overwrite_existing = overwrite)

    # A vertical slice running down the length of the fjord through the mooring.
    simulation.output_writers[:section] = NetCDFWriter(model, outputs;
        filename = joinpath(outdir, "$(prefix)_section.nc"),
        schedule = TimeInterval(config.section_interval),
        indices = (ix, :, :), overwrite_existing = overwrite)

    # Horizontal (map-view) slices at the surface and at the outflow depth.
    simulation.output_writers[:surface] = NetCDFWriter(model, outputs;
        filename = joinpath(outdir, "$(prefix)_surface.nc"),
        schedule = TimeInterval(config.section_interval),
        indices = (:, :, ksurf), overwrite_existing = overwrite)
    simulation.output_writers[:outflow] = NetCDFWriter(model, outputs;
        filename = joinpath(outdir, "$(prefix)_outflow.nc"),
        schedule = TimeInterval(config.section_interval),
        indices = (:, :, kout), overwrite_existing = overwrite)

    # Time-averaged fields, useful for the mean circulation.
    simulation.output_writers[:average] = NetCDFWriter(model, (; u, v, w, T, S);
        filename = joinpath(outdir, "$(prefix)_average.nc"),
        schedule = AveragedTimeInterval(config.average_interval, window = config.average_interval),
        overwrite_existing = overwrite)

    # Restart checkpoints (the full model state). `cleanup = true` keeps only the latest.
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

    # Pick the hardware: an explicit choice, or auto-detect a GPU and fall back to the CPU.
    arch = config.architecture == :cpu ? CPU() :
           config.architecture == :gpu ? GPU() :
           (has_cuda_gpu() ? GPU() : CPU())
    @info "Greenland fjord LES" name=config.name scenario=config.scenario architecture=arch

    grid = build_grid(config, arch)
    model, p = build_model(config, grid)
    set_initial_condition!(model, config, p)

    # A Simulation ties the model to a clock and a stopping time. We start with a small
    # timestep; the wizard below grows or shrinks it to stay just under the stability limit.
    simulation = Simulation(model; Δt = 0.5, stop_time = config.stop_days * days)
    simulation.callbacks[:wizard] = Callback(
        TimeStepWizard(cfl = config.cfl, max_change = 1.05, min_change = 0.2, max_Δt = config.max_dt),
        IterationInterval(2))
    simulation.callbacks[:progress] = Callback(log_progress, IterationInterval(20))

    pickup = add_outputs!(simulation, model, config)

    @info "starting run" stop_days=config.stop_days pickup
    run!(simulation; pickup)     # this line does the actual time-stepping
    @info "done: $(config.name)"
end

main()
