using NCDatasets
using CairoMakie
using Printf


# Read the vertical section saved by the simulation
# Use "output/control_section.nc" for your original test run instead.

# path = "output/control_movie_section.nc"
path = "output/control_noise_test_section.nc"

# The do-block closes the NetCDF file automatically when plotting is finished.
NCDataset(path) do ds
    variable = ds["T"] # temperature

    # Find which dimensions represent distance along the fjord, depth, andn time.
    # This avoids assuming they always appear in the same order in the file.
    dims = String.(dimnames(variable))
    println("Temperature dimensions: ", dims)

    yi = findfirst(d -> startswith(d, "y") || startswith(d, "j_"), dims)
    zi = findfirst(d -> startswith(d, "z") || startswith(d, "k_"), dims)
    ti = findfirst(d -> d == "time", dims)

    # Stop with a useful message if the file has an unexpected layout.
    isnothing(yi) && error("Could not identify the along-fjord dimension: $dims")
    isnothing(zi) && error("Could not identify the depth dimension: $dims")
    isnothing(ti) && error("Could not identify the time dimension: $dims")

    number_of_frames = size(variable, ti)
    # Read the saved simulation times in seconds.
    time_seconds = Float64.(ds["time"].var[:])
    println("Saved frames: ", number_of_frames)

    # Extract a two-dimensional temperature section at saved timestep n.
    function temperature_frame(n)
        # Select the first value of every dimension, fixing the across-fjord x position.
        indices = Any[1 for _ in dims]      # fix the across-fjord position
        
        # Keep all positions along the fjord and all depths.
        indices[yi] = Colon()               # keep every along-fjord point
        indices[zi] = Colon()               # keep every depth
        
        indices[ti] = n                     # select just one requested timestep
        
        values = variable[indices...]
        
        # Replace missing values, such as points below the seafloor, with NaN.
        # Makie leaves those points uncolored.
        data = Float64.(coalesce.(values, NaN))

        # Makie expects rows to follow the along-fjord coordinate.
        # Arrange the matrix so its first dimension is distance along the 
        # fjord and its second dimension is depth, as heatmap! expects.
        return yi < zi ? data : permutedims(data)
    end

    # Read one frame to determine how many distance and depth points exist. 
    first_frame = temperature_frame(1)
    
    # Make plot coordinates. The model fjord is 15 km long and 200 m deep.
    distance = range(0, 15; length = size(first_frame, 1))  #km
    depth = range(-200, 0; length = size(first_frame, 2))   #m

    # Create the figure and its labeled axes.
    figure = Figure(size=(1000, 500))
    title = Observable("Temperature - frame 1")  
    axis = Axis(figure[1,1];
        title = title,
        xlabel = "Distance from glacier (km)",
        ylabel = "Depth (m)",
    )

    # An Observable holds the data displayed by the heatmap.
    # Changing its value updates the plot for each movie frame.
    current_frame = Observable(first_frame)
    plot = heatmap!(
        axis, distance, depth, current_frame;
        colormap=:thermal, 
        colorrange=(1, 3),
    )

    # Show which colors correspond to which temperatures.
    Colorbar(figure[1, 2], plot; label = "Temperature (°C)")

    # Save a still image of the first timestep.
    # save("fjord_temperature.png", figure)

    save("control_noise_temperature.png", figure)

    # Create an MP4 by updating the temperature and title for each timestep.
    # At 6 frames per second, 36 saved timesteps make a 6-second movie. 
    # record(
    #     figure, "fjord_temperature.mp4", 1:number_of_frames;
    #     framerate = 6,
    #  ) do n
    #     current_frame[] = temperature_frame(n)
    #     title[] = "Temperature - frame $n of $number_of_frames"
    # end
    record(
        figure, "control_noise_temperature.mp4", 1:number_of_frames;
        framerate=6,
    ) do n
        current_frame[] = temperature_frame(n)
        title[] = @sprintf("Temperature — %.2f hours", time_seconds[n] / 3600)
    end

    println("Wrote fjord_temperature.png and fjord_temperature.mp4")

end







"""
Plot 
    - Vertical slice of simulated water temperature through the middle of the fjord
    - Horizontal axis - distance from glacier (0 km is glacier end, 15 km is fjord mouth)
    - Slice taken from middle of fjord's width, about x = 6 km
    - Vertical axis - depth, 0 m is the surface and -200 m is the deepest modeled level.
    - Color - temperature, the color bar maps temps from 1-3°C (warmer colors = warmer water, cooler = cooler water)

    At the start, there is a relatively warm layer near the surface and colder water below.
    This comes from the initial temperature profile in fjord.jl: about 3°C at the surface, 
        about 1°C at depth, with main transition around 15 m below the surface.
    Boundary between those layers is where an internal wave could become visible as the simulation evolves.

Movie
    Shows how the slice of water changes over the 1.2 hours of simulated time.
    Currently: 27 frames (two minutes of simualted time apart, through 72 minutes played at 6 FPS compresses into ~ 6 seconds)
   
    Glacier outflow is steady periodic tide at the mouth is off
    Model nudges water near the flacier toward an outflow centred around 15 m depth
    As water moves and mixes, colors shift, and boundary between warmer and colder water may balanced
    Bend in the temperature layers can indicate vertical displacement of those layers (temp movie does not prove an internal tide)

    make_bathymetry.jl put a raised sill across the fjord at 9 km from the glacier
    Bottom rises from ~200 m deep to ~80 m deep at the sill
    Plot may show blank/unusual-colored region beneath that bottom (that region's water temp should not be interepreted)
    
"""