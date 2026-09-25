# =====================================================================================
# make_bathymetry.jl - write an idealized fjord bottom to bottom.nc
#
# This is a stand-in so the model runs before you have real data. It builds a simple fjord
# shape: a U-shaped channel (deep in the middle, shoaling to the side walls) with a shallow
# sill part-way down. Run it once to create bottom.nc:
#
#   julia --project make_bathymetry.jl
#
# REPLACING IT WITH REAL BATHYMETRY (GEBCO)
# -----------------------------------------
# Download the GEBCO gridded bathymetry for your fjord's region. GEBCO stores a variable
# usually called "elevation" in metres, positive above sea level and negative below, which
# is already the "bottom height" convention this model uses (negative = below the surface).
# Subset it to your fjord, rotate/crop so the glacier is at y = 0 and the mouth at y = Ly,
# and save it as a NetCDF file. Then point fjord.jl at it by setting, in FjordConfig,
# `bathymetry_file` to your file and `bathymetry_var` to the depth variable's name
# (e.g. "elevation" for a raw GEBCO file). It need not match the model grid: fjord.jl
# resamples it automatically.
# =====================================================================================

using NCDatasets

# Domain matches the defaults in fjord.jl; the model resamples anyway if they differ.
Lx, Ly, Lz = 12_000.0, 15_000.0, 200.0    # width, length, depth [m]
nx, ny = 200, 240                          # bathymetry grid (any resolution is fine)

x = [(i - 0.5) * Lx / nx for i in 1:nx]    # cell-centre positions across the fjord [m]
y = [(j - 0.5) * Ly / ny for j in 1:ny]    # cell-centre positions along the fjord  [m]

sill_y      = 0.60 * Ly    # where along the fjord the sill sits [m]
sill_height = 120.0        # how far the sill rises off the deep bottom [m]
sill_width  = 1_200.0      # along-fjord half-width of the sill [m]

# Build the bottom height at every (x, y). Values are negative (below the surface).
bottom = Array{Float64}(undef, nx, ny)
for i in 1:nx, j in 1:ny
    r = (x[i] - Lx / 2) / (0.45 * Lx)             # 0 at the centre line, +/-1 near the walls
    channel = Lz * (1 - 0.9 * min(r^2, 1.0))      # deep in the middle, shallow at the walls
    sill = sill_height * exp(-((y[j] - sill_y) / sill_width)^2)   # a smooth ridge across the fjord
    depth = clamp(channel - sill, 10.0, Lz)       # keep at least 10 m of water everywhere
    bottom[i, j] = -depth                          # height convention: negative = below surface
end

# Write it to a NetCDF file with x, y coordinates and the "bottom" variable fjord.jl reads.
isfile("bottom.nc") && rm("bottom.nc")
NCDataset("bottom.nc", "c") do ds
    defDim(ds, "x", nx)
    defDim(ds, "y", ny)
    defVar(ds, "x", x, ("x",))
    defVar(ds, "y", y, ("y",))
    v = defVar(ds, "bottom", Float64, ("x", "y"))
    v[:, :] = bottom
    v.attrib["long_name"] = "bottom elevation (negative below sea level)"
    v.attrib["units"] = "m"
end

@info "wrote bottom.nc" size = (nx, ny) deepest = minimum(bottom) shallowest = maximum(bottom)
