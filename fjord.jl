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
#   Subglacial discharge:       RACMO 2.3p2 runoff (Zenodo 19254871) -> `discharge`. Plume
#                               theory (plume_theory, below) turns it, with the grounding-
#                               line depth and the ambient stratification, into the outflow
#                               depth, flux, temperature and salinity (see the plume block).
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
using TOML                         # read the fitted ambient profiles (standard library)

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
    min_depth::Float64      = 0.0      # water shallower than this [m] becomes land (0 = keep all);
                                         #   water cut off from the mouth always becomes land

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
    # Or use real casts instead of the tanh: point `ambient_file` at the ambient_profiles.toml
    # that files/omg_select_casts.py (via omg_profile_fit.py) writes - piecewise-polynomial
    # fits, still plain arithmetic, so still cheap on the GPU. One fitted cast fills the fjord
    # and feeds the plume theory; the other is what the mouth sponge relaxes toward.
    ambient_file::String     = ""            # "" = use the tanh profile above
    ambient_interior::String = "inside"      # cast for the fjord interior and the plume
    ambient_mouth::String    = "outside"     # cast for the mouth sponge

    # --- subglacial-discharge plume ---
    # Meltwater leaves the glacier bed at the grounding line, rises up the ice face as a
    # turbulent plume that entrains (mixes in) fjord water, and detaches where it becomes
    # neutrally buoyant. The grid cannot resolve that rising plume, so we solve it with
    # buoyant-plume theory (Morton, Taylor & Turner 1956, as reviewed in Straneo & Cenedese
    # 2015, Annu. Rev. Mar. Sci. 7:89, section 3.2; see `plume_theory` below) and inject only
    # its *outflow*: at the predicted neutral-buoyancy depth, with the predicted volume flux,
    # temperature and salinity, as a patch next to the glacier (y near 0).
    # Inputs: the discharge (RACMO 2.3p2 runoff, or e.g. the Slater et al. 2022 table:
    # Jakobshavn 1283 m^3/s summer mean) and the grounding-line depth.
    discharge::Float64            = 130.0    # subglacial discharge Q_sg [m^3/s]
    grounding_line_depth::Float64 = NaN      # [m, positive]; NaN = sea-floor depth at the plume
    plume_type::Symbol            = :point   # :point (one channel, eq. 7) | :line (eq. 8)
    entrainment::Float64          = 0.13     # entrainment coefficient alpha (top-hat profiles)
    T_discharge::Float64          = 0.0      # meltwater temperature [degC] (fresh, ~freezing)
    S_discharge::Float64          = 0.0      # meltwater salinity [g/kg]
    outflow_thickness::Float64    = 0.0      # outflow layer thickness [m]; 0 = plume diameter
    plume_x_center::Float64 = 6_000.0        # where across the fjord the plume enters [m]
    plume_x_width::Float64  = 200.0          # width of the outflow patch along the glacier wall [m];
                                             #   for :line plumes also the line-source length l
    plume_y_width::Float64  = 150.0          # how far the patch reaches out from the glacier wall [m]
                                             #   (widened to one grid cell if smaller)
    # Which wall the glacier is on. :y0 (default) is the glacier end of the box. A glacier at
    # the end of a side branch that meets the fjord at a right angle (Helheim, in Sermilik)
    # sits on a side wall instead: :x0 (x = 0) or :xL (x = Lx); the plume then enters at
    # y = plume_y_center and flows across the box (u), and plume_x_center is not used.
    plume_wall::Symbol       = :y0
    plume_y_center::Float64  = 0.0           # with a side wall: where along the box the plume enters [m]
    # How far the ice face is out from that wall [m]. 0 = on the wall. A glacier whose front is
    # inside the box (ice, i.e. land, between it and the wall, as for Helheim in a box that also
    # holds the shelf west of it) gets the distance make_bottom.py prints.
    plume_wall_offset::Float64 = 0.0
    relax_time::Float64     = 60.0           # how quickly we nudge the patch to target [s]

    # --- tide and discharge pulsing ---
    # M2 is the main lunar semidiurnal tide (one cycle every 12.42 h). In the :pump runs we
    # multiply the discharge by (1 + pump_amplitude * sin(2*pi*t/M2)) so it strengthens and
    # weakens each tidal cycle; plume theory says the outflow flux scales with the discharge
    # to the 1/3 power (eq. 7: Q ~ B^(1/3)), so the outflow swings by (1 + ...)^(1/3).
    # In the :tide runs we push a weak back-and-forth flow at the mouth to represent the
    # basin tide reaching the fjord.
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

# With an `ambient_file`, p.casts holds the fitted casts and these use them instead: a
# polynomial above the break depth, another below, with z held inside the cast's depth range
# (so below the deepest measurement the profile stays at its deepest value). The method is
# picked by the type of p.casts (nothing = tanh), so the GPU never branches on it.
@inline piecewise(z, f) = (zc = clamp(z, f.zmin, zero(z)); zc > f.zbreak ? evalpoly(zc, f.up) : evalpoly(zc, f.dn))

@inline ambient_T(z, p) = ambient_T(z, p, p.casts)       # the fjord interior (and the plume)
@inline ambient_S(z, p) = ambient_S(z, p, p.casts)
@inline mouth_T(z, p)   = mouth_T(z, p, p.casts)         # what the mouth sponge relaxes toward
@inline mouth_S(z, p)   = mouth_S(z, p, p.casts)
@inline ambient_T(z, p, ::Nothing) = p.T_deep + (p.T_surface - p.T_deep) * surface_fraction(z, p)
@inline ambient_S(z, p, ::Nothing) = p.S_deep + (p.S_surface - p.S_deep) * surface_fraction(z, p)
@inline mouth_T(z, p, ::Nothing)   = ambient_T(z, p, nothing)
@inline mouth_S(z, p, ::Nothing)   = ambient_S(z, p, nothing)
@inline ambient_T(z, p, c) = piecewise(z, c.interior.T)
@inline ambient_S(z, p, c) = piecewise(z, c.interior.S)
@inline mouth_T(z, p, c)   = piecewise(z, c.mouth.T)
@inline mouth_S(z, p, c)   = piecewise(z, c.mouth.S)

# Read the fitted casts from `ambient_file` (nothing if it is empty). Coefficients are stored
# highest power first; evalpoly wants them lowest first, as a tuple (a plain, GPU-safe value).
function load_casts(config)
    isempty(config.ambient_file) && return nothing
    path = isabspath(config.ambient_file) ? config.ambient_file : joinpath(@__DIR__, config.ambient_file)
    d = TOML.parsefile(path)
    function fit(label, var)
        haskey(d, label) || error("no cast '$label' in $path (it has: $(join(keys(d), ", ")))")
        f = d[label][var]
        return (zmin = Float64(f["zmin"]), zbreak = Float64(f["zbreak"]),
                up = Tuple(Float64.(reverse(f["up"]))), dn = Tuple(Float64.(reverse(f["dn"]))))
    end
    casts = (interior = (T = fit(config.ambient_interior, "T"), S = fit(config.ambient_interior, "S")),
             mouth    = (T = fit(config.ambient_mouth, "T"),    S = fit(config.ambient_mouth, "S")))
    @info "ambient profiles from $(basename(path))" interior = config.ambient_interior mouth = config.ambient_mouth
    return casts
end

# The numbers the ambient functions need, bundled (shared by the plume theory and the forcing).
ambient_params(config) = (pycnocline_depth = config.pycnocline_depth, pycnocline_scale = config.pycnocline_scale,
                          T_surface = config.T_surface, T_deep = config.T_deep,
                          S_surface = config.S_surface, S_deep = config.S_deep, casts = load_casts(config))

# The outflow speed as a function of depth: zero outside the injection band, rising to a
# peak of `outflow_speed` at the neutral-buoyancy depth (-outflow_depth), and tapering
# linearly to zero at the top and bottom of the band. Picture a tent whose peak sits at the
# outflow depth. (When the plume reaches the surface the peak sits at the band's top.)
@inline function outflow_velocity(z, p)
    zc = -p.outflow_depth
    if z > p.source_top || z < p.source_bottom
        return zero(z)                                   # outside the band: no injection
    elseif z >= zc
        return p.source_top > zc ? p.outflow_speed * (p.source_top - z) / (p.source_top - zc) :
                                   p.outflow_speed                            # peak -> 0 upward
    else
        return p.outflow_speed * (z - p.source_bottom) / (zc - p.source_bottom)  # 0 -> peak
    end
end

# -------------------------------------------------------------------------------------
# 2b. Plume theory: from discharge to outflow
# -------------------------------------------------------------------------------------
# A buoyant plume rising from a source of fresh water entrains surrounding water at a rate
# proportional to its own speed (the "entrainment assumption": inflow speed = alpha * w).
# Conserving volume, momentum, heat and salt in the plume gives, per metre of rise
# (Straneo & Cenedese 2015, section 3.2.1; Morton et al. 1956):
#
#   point source (one channel), radius b:   dQ/dz = 2 pi b alpha w,   d(Qw)/dz = pi b^2 g'
#   line source (a slot of length l), half-width b, per metre of l:
#                                           dQ/dz = 2 alpha w,        d(Qw)/dz = 2 b g'
#   both:                                   d(QT)/dz = (dQ/dz) T_ambient, same for S
#
# with g' = g [alpha_T (T - T_a) - beta_S (S - S_a)] the plume's reduced gravity, computed
# with the same linear equation of state the model uses. In uniform water these have the
# self-similar solutions of the paper's eqs. (7) and (8); we use those to start the plume
# from the grounding line (matching its volume flux to the discharge, i.e. a virtual origin
# below the source) and then integrate numerically up through the real stratification.
# The plume detaches where g' first reaches 0 - the neutral-buoyancy (intrusion) depth of
# section 3.2.2 - or at the surface if it is still buoyant there.
# Runs once, on the CPU, before the simulation.
function plume_theory(config, z_gl)
    g, α = 9.80665, config.entrainment
    αT, βS = config.thermal_expansion, config.haline_contraction
    amb = ambient_params(config)
    reduced_gravity(T, S, z) = g * (αT * (T - ambient_T(z, amb)) - βS * (S - ambient_S(z, amb)))
    point = config.plume_type == :point
    point || config.plume_type == :line || error("plume_type must be :point or :line")

    # Source: pure meltwater at the grounding line. For a line plume, fluxes are per metre of l.
    l = config.plume_x_width
    Q = point ? config.discharge : config.discharge / l
    T, S = config.T_discharge, config.S_discharge
    g0 = reduced_gravity(T, S, z_gl)
    g0 > 0 || error("the discharge is not lighter than the ambient water at the grounding line")
    B = g0 * Q                                                   # buoyancy flux
    if point                                                     # eq. (7)
        cQ = 6 / 5 * (9 / 10)^(1 / 3) * π^(2 / 3) * α^(4 / 3) * B^(1 / 3)
        z0 = (Q / cQ)^(3 / 5)                                    # height where Q(z0) = Q_sg
        w = 5 / (6α) * (9 / 10 * α * B)^(1 / 3) * π^(-1 / 3) * z0^(-1 / 3)
    else                                                         # eq. (8)
        w = (2α)^(-1 / 3) * B^(1 / 3)
    end
    M, FT, FS = Q * w, Q * T, Q * S                              # momentum, heat, salt fluxes

    function tendencies(Q, M, FT, FS, z)
        w = M / Q
        gp = reduced_gravity(FT / Q, FS / Q, z)
        E = point ? 2α * sqrt(π * Q * w) : 2α * w                # entrainment (dQ/dz)
        return E, Q * gp / w, E * ambient_T(z, amb), E * ambient_S(z, amb)
    end

    # March upward in small steps (midpoint rule) until the plume reaches the surface or runs
    # out of momentum, recording the first neutral-buoyancy crossing.
    dz = 0.05
    z = z_gl
    nb = nothing
    gp_prev = g0
    while z < 0 && M > 0
        k1 = tendencies(Q, M, FT, FS, z)
        h = min(dz, -z)
        Qm, Mm = Q + h / 2 * k1[1], M + h / 2 * k1[2]
        Mm > 0 || break
        k2 = tendencies(Qm, Mm, FT + h / 2 * k1[3], FS + h / 2 * k1[4], z + h / 2)
        Q, M, FT, FS = Q + h * k2[1], M + h * k2[2], FT + h * k2[3], FS + h * k2[4]
        z += h
        M > 0 || break
        gp = reduced_gravity(FT / Q, FS / Q, z)
        if nb === nothing && gp_prev > 0 && gp <= 0
            nb = (; z, Q, w = M / Q, T = FT / Q, S = FS / Q)
        end
        gp_prev = gp
    end
    surfaced = nb === nothing
    nb = surfaced ? (; z = min(z, 0.0), Q, w = M / Q, T = FT / Q, S = FS / Q) : nb
    b = point ? sqrt(nb.Q / (π * nb.w)) : nb.Q / (2nb.w)        # radius / half-width there
    Qtot = point ? nb.Q : nb.Q * l                               # total volume flux [m^3/s]
    return (; z_nb = nb.z, z_top = min(z, 0.0), Q = Qtot, T = nb.T, S = nb.S, w = nb.w, b,
              surfaced, buoyancy_flux = point ? B : B * l, g0)
end

# -------------------------------------------------------------------------------------
# 3. Source and sponge regions (the "forcing" that drives the fjord)
# -------------------------------------------------------------------------------------
# Oceananigans lets us add a forcing term to each equation: an extra tendency evaluated at
# every point and time. We use it two ways: (1) inject the plume near the glacier, and
# (2) sponge the flow near the mouth so water can leave and T/S stay realistic.

# True inside the small box near the glacier where we inject the plume outflow.
# p.wall: 0 = glacier at y = 0, 1 = at x = 0, 2 = at x = Lx (see plume_wall).
@inline function in_source(x, y, z, p)
    along = p.wall == 0 ? x : y                                   # position along the glacier wall
    out   = p.wall == 0 ? y : p.wall == 1 ? x : p.Lx - x          # distance out from the wall
    return (abs(along - p.plume_center) <= 0.5 * p.source_x_width) &&  # within the patch along the wall
           (p.wall_offset <= out <= p.wall_offset + p.source_y_width) &&  # just off the ice face
           (z >= p.source_bottom) && (z <= p.source_top)               # within the injection depth band
end

# The outflow speed the source nudges toward: the tent profile, pulsing with the discharge
# by (1 + A sin)^(1/3) (plume theory, eq. 7; 1.0 in steady runs).
@inline outflow_target(z, t, p) =
    outflow_velocity(z, p) * cbrt(1 + p.pump_amplitude * sin(2π * t / p.M2_period))

# A ramp that is 0 at the start of the mouth sponge and grows to 1 at the far wall, so the
# nudging turns on gradually (a sudden wall of forcing would bounce waves back).
@inline mouth_ramp(y, p)     = y >= p.mouth_start ? (y - p.mouth_start) / (p.Ly - p.mouth_start) : zero(y)
@inline mouth_deep(y, z, p)  = z < -5 ? mouth_ramp(y, p) : zero(y)          # below the surface layer
@inline mouth_upper(y, z, p) = (-20 <= z <= 0) ? mouth_ramp(y, p) : zero(y) # the upper outflow layer

# Forcing on the along-fjord velocity v. Three additive pieces:
#   src : inside the source box (glacier at y = 0), nudge v toward the plume outflow speed.
#   out : near the mouth's upper layer, nudge v toward a slow steady outflow (mass balance).
#   tide: near the mouth's deeper layer, nudge v toward a back-and-forth tidal current.
@inline function force_v(x, y, z, t, v, p)
    src  = (p.wall == 0 && in_source(x, y, z, p)) ? -(v - outflow_target(z, t, p)) / p.relax_time : zero(v)
    out  = -mouth_upper(y, z, p) / p.sponge_time * (v - p.mouth_outflow)
    tide = -mouth_deep(y, z, p)  / p.tide_sponge_time * (v - p.tide_amplitude * sin(2π * t / p.M2_period))
    return src + out + tide
end

# Forcing on the across-fjord velocity u: only for a glacier on a side wall, where the plume
# outflow leaves the wall across the box (+u from x = 0, -u from x = Lx).
@inline function force_u(x, y, z, t, u, p)
    sgn = p.wall == 1 ? one(u) : -one(u)
    return in_source(x, y, z, p) ? -(u - sgn * outflow_target(z, t, p)) / p.relax_time : zero(u)
end

# Forcing on temperature: inject the plume's temperature in the source box, and relax back
# to the ambient profile near the mouth. p.T_source is the plume outflow's temperature from
# plume theory: meltwater mixed with all the fjord water it entrained on the way up.
@inline function force_T(x, y, z, t, T, p)
    src = in_source(x, y, z, p) ? -(T - p.T_source) / p.relax_time : zero(T)
    return src - mouth_deep(y, z, p) / p.sponge_time * (T - mouth_T(z, p))
end

# Forcing on salinity: the same idea. Plume theory makes the outflow's T and S exactly
# neutrally buoyant at the outflow depth, so it stays there instead of rising or sinking.
@inline function force_S(x, y, z, t, S, p)
    src = in_source(x, y, z, p) ? -(S - p.S_source) / p.relax_time : zero(S)
    return src - mouth_deep(y, z, p) / p.sponge_time * (S - mouth_S(z, p))
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
        bottom = B                                       # already on our grid: use as-is
    else
        # Otherwise interpolate the file's grid onto our model grid. This is why you can
        # feed in raw GEBCO at its own resolution and still run any Nx, Ny you like.
        nbx, nby = size(B)
        itp = linear_interpolation(
            (range(0, config.Lx; length = nbx), range(0, config.Ly; length = nby)),
            B; extrapolation_bc = Flat())
        xc = [(i - 0.5) * config.Lx / config.Nx for i in 1:config.Nx]   # cell-centre positions
        yc = [(j - 0.5) * config.Ly / config.Ny for j in 1:config.Ny]
        bottom = [itp(xc[i], yc[j]) for i in 1:config.Nx, j in 1:config.Ny]
        @warn "bathymetry $(size(B)) resampled onto the ($(config.Nx), $(config.Ny)) grid"
    end
    bottom = clean_bottom(bottom, config.min_depth)

    # GridFittedBottom switches off the cells that lie below `bottom`, giving the fjord its shape.
    return ImmersedBoundaryGrid(underlying, GridFittedBottom(on_architecture(arch, bottom)))
end

# Tidy the bottom on the model grid (after any resampling, which can make new shallow cells
# and cut-off pockets along the coast): columns shallower than `min_depth` become land (thin
# coastal strips are hard on the pressure solver), and water not connected through cell faces
# to the mouth row (y = Ly) becomes land too - lakes and cut-off bays the fjord cannot reach.
function clean_bottom(B, min_depth)
    B = copy(B)
    nx, ny = size(B)
    shallow = (-min_depth .< B .< 0)
    B[shallow] .= 0
    reach = falses(nx, ny)                       # flood-fill the water from the mouth row
    stack = [(i, ny) for i in 1:nx if B[i, ny] < 0]
    foreach(((i, j),) -> reach[i, j] = true, stack)
    while !isempty(stack)
        i, j = pop!(stack)
        for (ii, jj) in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
            if 1 <= ii <= nx && 1 <= jj <= ny && !reach[ii, jj] && B[ii, jj] < 0
                reach[ii, jj] = true
                push!(stack, (ii, jj))
            end
        end
    end
    any(reach) || error("no water at the mouth (y = Ly) in the bathymetry")
    cut = (B .< 0) .& .!reach
    B[cut] .= 0
    (any(shallow) || any(cut)) && @info "bathymetry tidied: set to land" min_depth shallow_cells = count(shallow) cut_off_cells = count(cut)
    return B
end

# -------------------------------------------------------------------------------------
# 5. Model
# -------------------------------------------------------------------------------------
# Turn the plume-theory outflow into the source patch the model nudges toward:
#   depth band  centred on the neutral-buoyancy depth, as thick as the plume is wide there
#               (or `outflow_thickness`), kept below the surface and at least two cells thick;
#   patch       `plume_x_width` along the glacier wall (or the plume diameter, if wider) by
#               `plume_y_width` out from it, each widened to whole grid cells so the source is
#               never empty;
#   speed       the tent-profile peak that carries the theory's volume flux through the patch
#               (a tent's area is peak x thickness / 2).
function plume_source(config, grid)
    dx, dy, dz = config.Lx / config.Nx, config.Ly / config.Ny, config.Lz / config.Nz
    config.plume_wall in (:y0, :x0, :xL) || error("plume_wall must be :y0, :x0 or :xL")
    side = config.plume_wall != :y0
    # along the glacier wall: spacing, cell count, length, plume position; and the spacing out from it
    d_along, n_along, L_along = side ? (dy, config.Ny, config.Ly) : (dx, config.Nx, config.Lx)
    center = side ? config.plume_y_center : config.plume_x_center
    d_out, n_out = side ? (dx, config.Nx) : (dy, config.Ny)
    # first cell out from the ice face (the wall itself unless plume_wall_offset is set)
    j = clamp(floor(Int, config.plume_wall_offset / d_out) + 1, 1, n_out)

    # grounding line: given, or the sea floor under the plume at the glacier face
    if isnan(config.grounding_line_depth)
        hb = Array(interior(grid.immersed_boundary.bottom_height))
        k = nearest_index(center, L_along, n_along)
        z_gl = Float64(config.plume_wall == :y0 ? hb[k, j, 1] :
                       config.plume_wall == :x0 ? hb[j, k, 1] : hb[config.Nx + 1 - j, k, 1])
        z_gl < 0 || error("no water under the plume at the glacier face - move the plume")
    else
        z_gl = -config.grounding_line_depth
    end

    pl = plume_theory(config, z_gl)

    thickness = config.outflow_thickness > 0 ? config.outflow_thickness : max(2pl.b, 2dz)
    top = min(pl.z_nb + thickness / 2, 0.0)
    bottom = top - thickness
    depth = -clamp(pl.z_nb, bottom, top)

    # count the grid columns the patch actually catches, so the injected flux is right
    width = config.plume_type == :line ? config.plume_x_width : max(config.plume_x_width, 2pl.b)
    ncols = count(i -> abs((i - 0.5) * d_along - center) <= width / 2, 1:n_along)
    if ncols == 0
        @warn "outflow patch ($(round(width)) m) is narrower than a grid cell ($(round(d_along)) m); widening to one cell"
        width, ncols = d_along, 1
    end
    y_width = config.plume_y_width
    if y_width < d_out
        @warn "plume_y_width ($(y_width) m) is shorter than a grid cell ($(round(d_out, digits = 1)) m); widening to one cell"
        y_width = d_out
    end
    speed = 2pl.Q / (ncols * d_along * thickness)

    @info "plume theory (Straneo & Cenedese 2015, sec. 3.2)" plume_type = config.plume_type discharge_m3s = config.discharge grounding_line_m = round(-z_gl, digits = 1) neutral_buoyancy_depth_m = round(-pl.z_nb, digits = 1) reaches_surface = pl.surfaced max_rise_to_m = round(-pl.z_top, digits = 1) outflow_flux_m3s = round(pl.Q, digits = 1) dilution = round(pl.Q / config.discharge, digits = 1) outflow_T = round(pl.T, digits = 3) outflow_S = round(pl.S, digits = 3) plume_radius_m = round(pl.b, digits = 1)
    @info "outflow source" band_m = (round(-top, digits = 1), round(-bottom, digits = 1)) patch_m = (round(ncols * d_along), round(y_width)) wall = config.plume_wall wall_offset_m = config.plume_wall_offset peak_speed_ms = round(speed, digits = 3)

    return (; x_width = width, y_width, speed, depth, top, bottom, T = pl.T, S = pl.S, theory = pl)
end

# Assemble the physics: advection, rotation, buoyancy, the pressure solver, the forcing,
# and the boundary conditions. This is where all the pieces come together into one model.
function build_model(config, grid)
    pump_on = config.scenario in (:pump, :tidepump)   # is the discharge pulsed?
    tide_on = config.scenario in (:tide, :tidepump)   # is the mouth tide on?

    src = plume_source(config, grid)

    # Bundle the scalars the forcing needs into one lightweight, GPU-friendly package.
    # (It holds only plain numbers, which is what lets it be used inside GPU kernels.)
    p = (; ambient_params(config)...,
         wall = config.plume_wall == :y0 ? 0 : config.plume_wall == :x0 ? 1 : 2, Lx = config.Lx,
         wall_offset = config.plume_wall_offset,
         plume_center = config.plume_wall == :y0 ? config.plume_x_center : config.plume_y_center,
         source_x_width = src.x_width,
         source_y_width = src.y_width, outflow_speed = src.speed,
         outflow_depth = src.depth, source_top = src.top, source_bottom = src.bottom,
         T_source = src.T, S_source = src.S, relax_time = config.relax_time,
         Ly = config.Ly, mouth_start = config.mouth_start, mouth_outflow = config.mouth_outflow,
         sponge_time = config.sponge_time, tide_sponge_time = config.tide_sponge_time,
         M2_period = config.M2_period,
         # Turn pumping/tide off (amplitude 0) unless this scenario calls for them.
         pump_amplitude = pump_on ? config.pump_amplitude : 0.0,
         tide_amplitude = tide_on ? config.tide_amplitude : 0.0)

    # Attach each forcing function to its field. `field_dependencies` tells Oceananigans
    # which field to pass in (so force_v receives the current v, etc.).
    forcing = (v = Forcing(force_v; parameters = p, field_dependencies = :v),
               T = Forcing(force_T; parameters = p, field_dependencies = :T),
               S = Forcing(force_S; parameters = p, field_dependencies = :S))
    if config.plume_wall != :y0          # glacier on a side wall: the outflow is pushed along x
        forcing = (; forcing..., u = Forcing(force_u; parameters = p, field_dependencies = :u))
    end

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
    # Each component has its own shape (u, v, w sit on x, y, z cell faces), so size each separately.
    noise(f) = 5e-3 .* (rand(size(f)...) .- 0.5)    # tiny random velocities, +/-2.5 mm/s
    uᵢ, vᵢ, wᵢ = noise(u), noise(v), noise(w)
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
function add_outputs!(simulation, model, config, p)
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
    kout  = nearest_index(config.Lz - p.outflow_depth, config.Lz, config.Nz)  # plume outflow depth

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

    pickup = add_outputs!(simulation, model, config, p)

    @info "starting run" stop_days=config.stop_days pickup
    run!(simulation; pickup)     # this line does the actual time-stepping
    @info "done: $(config.name)"
end

# Run only when executed as a script, so `include("fjord.jl")` can load the functions for testing.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
