# greenland-fjords

An idealized, internal-wave-permitting large-eddy simulation of a glacial fjord, built on
[Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl). A subglacial-discharge plume
enters at the head of the fjord and the model resolves the circulation and internal tide it
drives. It is a general, stripped-down version of a Saqqarleq (West Greenland) configuration:
all site-specific numbers live in one configuration block, and the bathymetry is read from a
NetCDF file, so the same code runs for any fjord.

## What it does

The model is a rotating, nonhydrostatic Boussinesq LES on a closed domain (solid walls plus a
grid-fitted immersed bottom). The glacier sits at `y = 0` and the fjord mouth at `y = Ly`. The
subglacial discharge is imposed as an interior relaxation source near the glacier face rather
than resolved at the grounding line: the source water is given the ambient density of the
outflow depth, so it is denser than the fresh surface layer, settles to that depth, and (when
the discharge is pulsed) radiates an internal tide. A sponge at the mouth returns the volume
flux so the closed domain stays balanced, which is what the conjugate-gradient pressure solver
requires.

Four scenarios isolate what drives the internal tide:

| scenario    | discharge        | barotropic tide |
|-------------|------------------|-----------------|
| `control`   | steady           | none            |
| `tide`      | steady           | M2 at the mouth |
| `pump`      | pulsed at M2     | none            |
| `tidepump`  | pulsed at M2     | M2 at the mouth |

## Numerics

Fifth-order WENO advection (the implicit LES closure, so no explicit turbulence closure),
a third-order Runge-Kutta timestepper with an adaptive step, a linear equation of state, an
f-plane, and a `ConjugateGradientPoissonSolver` that enforces the immersed no-normal-flow
condition inside the pressure projection (this removes a spurious near-bottom tracer signal
that the default Fourier solver produces over grid-fitted bathymetry).

## Install

Requires Julia 1.10+. From the repo directory:

```julia
julia --project -e 'using Pkg; Pkg.instantiate()'
```

A CUDA-capable GPU is used automatically when available; otherwise the model runs on the CPU.

## Quickstart

```bash
# 1. generate an idealized bathymetry (writes bottom.nc)
julia --project make_bathymetry.jl

# 2. a fast CPU smoke test on a coarse grid (~1 hour of model time)
julia --project fjord.jl --scenario=control --architecture=cpu \
      --Nx=64 --Ny=80 --Nz=40 --stop_days=0.05

# 3. a full GPU run
julia --project fjord.jl --scenario=tidepump --architecture=gpu --stop_days=7
```

Any field in `FjordConfig` can be overridden on the command line as `--field=value`; for
example `--Lx=8000 --outflow_depth=20 --tide_amplitude=0.02`. Everything else is edited in the
configuration block at the top of `fjord.jl`.

## Configuration

`fjord.jl` opens with a single `FjordConfig` struct grouping every parameter: domain and grid,
the bathymetry file and variable name, physics constants, the ambient stratification (a
fresh/warm surface layer over cool/salty deep water, blended with a tanh; swap in a fit to a
real cast), the plume source geometry and strength, the tide and pulsing amplitudes, the mouth
sponge, the virtual mooring location, output cadence, and numerics. Each field is commented
with its units.

## Bathymetry

`bottom.nc` holds a 2-D bottom-elevation array under the variable named by
`bathymetry_var` (default `bottom`). It does not need to match the model grid: if the shapes
differ it is linearly resampled onto the grid. `make_bathymetry.jl` writes an idealized
U-shaped channel with a sill; replace it with a real fjord's bathymetry to model a specific
site.

## Outputs

Written to `output/` (or `--output_dir`), one NetCDF file per stream, prefixed by the run
`--name`:

- `*_mooring.nc` — full-depth column at the virtual mooring, high cadence
- `*_section.nc` — along-fjord section through the mooring
- `*_surface.nc`, `*_outflow.nc` — horizontal slices at the surface and the outflow depth
- `*_average.nc` — time-averaged fields
- `checkpoint_*` — restart checkpoints; rerunning the same `--name`/`--output_dir` resumes

## Running on NCAR Casper

`submit_casper.sh` launches one scenario on an A100 and resumes from a checkpoint if present:

```bash
qsub -v SCENARIO=control                      submit_casper.sh
qsub -v SCENARIO=tidepump,EXTRA=--stop_days=7 submit_casper.sh
```

## License

MIT (see `LICENSE`).
