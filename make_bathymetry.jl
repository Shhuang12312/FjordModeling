# =====================================================================================
# make_bathymetry.jl — write an idealized fjord bottom to bottom.nc
#
# Produces a U-shaped channel (deep in the middle, shoaling to the side walls) with a
# sill partway down the fjord, so the repo runs out of the box. Replace bottom.nc with a
# real fjord's bathymetry (same variable name, any resolution) when you have one.
#
#   julia --project make_bathymetry.jl
# =====================================================================================

using NCDatasets

# Match the defaults in fjord.jl; the model resamples if the grid differs anyway.
Lx, Ly, Lz = 12_000.0, 15_000.0, 200.0
nx, ny = 200, 240

x = [(i - 0.5) * Lx / nx for i in 1:nx]
y = [(j - 0.5) * Ly / ny for j in 1:ny]

sill_y      = 0.60 * Ly    # sill location along the fjord [m]
sill_height = 120.0        # how far the sill rises off the deep bottom [m]
sill_width  = 1_200.0      # along-fjord half-width of the sill [m]

bottom = Array{Float64}(undef, nx, ny)
for i in 1:nx, j in 1:ny
    r = (x[i] - Lx / 2) / (0.45 * Lx)                 # 0 at centre, ±1 near the walls
    channel = Lz * (1 - 0.9 * min(r^2, 1.0))          # deep channel, shallow walls
    sill = sill_height * exp(-((y[j] - sill_y) / sill_width)^2)
    depth = clamp(channel - sill, 10.0, Lz)
    bottom[i, j] = -depth                             # bottom elevation (negative, below surface)
end

isfile("bottom.nc") && rm("bottom.nc")
NCDataset("bottom.nc", "c") do ds
    defDim(ds, "x", nx)
    defDim(ds, "y", ny)
    defVar(ds, "x", x, ("x",))
    defVar(ds, "y", y, ("y",))
    v = defVar(ds, "bottom", Float64, ("x", "y"))
    v[:, :] = bottom
    v.attrib["long_name"] = "bottom elevation"
    v.attrib["units"] = "m"
end

@info "wrote bottom.nc" size = (nx, ny) deepest = minimum(bottom) shallowest = maximum(bottom)
