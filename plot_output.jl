# plot_output.jl
using ArgParse
using JLD2
using CairoMakie
using Oceananigans

function parse_command_line()
    s = ArgParseSettings(description="Plot FjordModeling simulation output.")
    @add_arg_table! s begin
        "--dir", "-d"
            help = "Output directory containing simulation files"
            default = "output_ilulissat_smoke"
        "--output", "-o"
            help = "Output image filename"
            default = "plots/simulation_output.png"
    end
    return parse_args(s)
end

function main()
    args = parse_command_line()
    out_dir = args["dir"]
    
    println("Looking for output files in: $out_dir")
    
    if !isdir(out_dir)
        error("Output directory '$out_dir' does not exist.")
    end
    
    # Find JLD2 output files in the directory
    jld2_files = filter(f -> endswith(f, ".jld2"), readdir(out_dir, join=true))
    
    if isempty(jld2_files)
        error("No .jld2 output files found in $out_dir.")
    end
    
    # Pick the first or latest output file
    file_path = jld2_files[1]
    println("Loading data from: $file_path")
    
    # Open the file and inspect available fields/timeseries
    # Replace this line:
# file = jld2open(file_path, "r")

# With this line:
    file = jldopen(file_path, "r")
    println("Available keys in file: ", keys(file))
    
    # Check if 'timeseries' exists in the JLD2 file
    if haskey(file, "timeseries")
        timeseries_keys = keys(file["timeseries"])
        println("Available fields in timeseries: ", timeseries_keys)
        
        # Example: Load the last available time step for a field (e.g., 'w' or 'b' or 'smoke' if saved)
        # Adjust field name based on what your model saves
        field_name = :w # change to your desired field, e.g., :u, :v, :w
        if string(field_name) in timeseries_keys || haskey(file["timeseries"], string(field_name))
            iterations = keys(file["timeseries/t"])
            latest_iter = last(iterations)
            println("Plotting field '$field_name' at iteration $latest_iter...")
        end
    end
    
    close(file)
    
    # Create output directory for plots if it doesn't exist
    mkpath(dirname(args["output"]))
    
    # Generate a simple visualization using CairoMakie
    fig = Figure(size = (900, 600))
    ax = Axis(fig[1, 1], title = "Fjord Simulation Output ($out_dir)", 
              xlabel = "x [m]", ylabel = "z [m]")
    
    # Placeholder text/plot elements until custom fields are sliced
    text!(ax, 0.5, 0.5, text = "Data loaded successfully!\nConfigure your slice plots here.", 
          align = (:center, :center), textsize = 20)
    
    save(args["output"], fig)
    println("Plot saved successfully to: $(args["output"])")
end

main()