# Animation functions
''' For making opendrift animations, open the dataset with xarray and then run the drift_frames function
    setting IBM={your opendrift dataset}. You can also change the extent that is plotted by passing
    domain={yourdomain}. Most importantly, set the directory where you want the frames to be saved
    using framedir={your frame directory}.
    
    After drift_frames has finished, run drift_animation, specifying your frame directory and where
    you want the animation to be saved to in the function call.
    
    The functions should work for opendrift datasets without the need for the other functions and specifics
    used for including SVP or other real drifter trajectories, but I have included the functions here in case
    python complains that they do not exist'''

# Required Libs
import numpy as np
import xarray as xr
from datetime import datetime
from datetime import timedelta
# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import cmocean
import matplotlib.animation as animation # For animations (not used atm)
from PIL import Image # For animations
import glob


def get_traj_index(ds) -> np.array: # IGNORE THIS FUNCTION IF JUST USING OPENDRIFT SIMS
    ''' Short function used to find the indexes in the obs dimension where
        drifter trajectories start. The function assumes that rowsizes
        have already been calculated, if not, pass the ds through the
        get_rowsize function
    Args:
        ds: ragged array dataset of drifter trajectories and obs
    Returns:
        An array of length equal to the length of the traj dim, where each
        value repressents where traj_n starts in the obs dimension
        AND an array with the length shortened by 1 (trajsum) for slicing'''
    
    # Sums the rowsizes to find indexes of each drifter (i.e., index for drifter n is rowsize of drifter n-1 + 1)
    trajsum = np.cumsum(ds.rowsize.values, dtype=np.int64)

    # Adds a zero at the start (repressents first drifter index)
    traj_index_full = np.insert(trajsum,0,0)

    # Deletes last index to re-align size
    traj_idx = np.delete(traj_index_full, len(traj_index_full)-1)
    return traj_idx, trajsum


def traj_cmap(traj, random_seed: int = 42): # IGNORE THIS FUNCTION IF JUST USING OPENDRIFT SIMS
    ''' Short function to create a coloUr map based on the length of a ds'''

    # Find unique IDs just in case there are duplicates
    unique_traj_ids = np.unique(traj)
    num_unique_trajectories = len(unique_traj_ids)
        
    # Generate a consistent set of colors based on trajectory IDs
    np.random.seed(random_seed) # Seed number is what makes it consistent
    random_colors = np.random.rand(num_unique_trajectories, 3)
        
    # Create a colormap using the consistent colors
    cmap = colors.ListedColormap(random_colors)
    norm = colors.BoundaryNorm(range(len(traj) + 1), cmap.N)
    return cmap, norm 


def drift_frames(ds: any = None, IBM: any = None, IBM_2: any = None, domain: any = (145,165,-45,-20),
                 plot_int: int = 1, ts: int = 1, labs: any = ['SVP drifters', 'Simulated drifters'],
                 framedir: str = 'C:/Users/z5493451/OneDrive - UNSW/Documents/Data/Drifters/animations',
                 duration: int = None, traj_cmap: any = None, traj_norm: any = None):
    '''Function to create frames for animations of drifter trajectories. Works for both GDP and IBM datasets.
        If both a real (i.e., GDP) and IBM dataset are provided, both will be plotted. If IBM datasets are
        provided, they will both be plotted. 
        
        Args:
            framedir: path and name of animation file to save to
            plot_int: interval in hours to plot positions
            labs: the labels used for each trajectory type
            plot_int: resolution of the animation (at what interval should frames be saved)
            ts: timestep (in hours) of the dataset (should usually be kept at 1 hour for GDP animations)
            duration: duration of the animation (in days)'''

    # Extract latitude / longitude from datasets
    if ds is not None:
        # Create a trajectory index
        traj_i, trajsums = get_traj_index(ds)
        # Split the lat and lon vars into trajectories and combine into lat/lon pairs
        trajectories = [
            {
                "lat_lon_pairs": np.column_stack((
                    ds.latitude.values[slice(traj_i[i], trajsums[i])],
                    ds.longitude.values[slice(traj_i[i], trajsums[i])])
                ),
                "timestamps": ds.dt.values[slice(traj_i[i], trajsums[i])],  # Add corresponding timestamps
                "ID": ds.ids.values[slice(traj_i[i], trajsums[i])],
                "color": ds.traj_cols.values[slice(traj_i[i], trajsums[i])]  # Add color based on trajectory number
            }
            for i in range(len(ds.traj))
        ]
        # Extract trajectory IDs
        trajectory_identifiers = ds.ID.values
    else:
        print('No real drifter dataset supplied, mapping IBM trajectories only.')
        
    if IBM is not None:
        trajectories_b = [
            {
                "lat_lon_pairs": np.column_stack((
                    (IBM.isel(trajectory=i).lat.values,
                     IBM.isel(trajectory=i).lon.values))
                ),
                "timestamps": IBM.isel(trajectory=i).age_seconds.values,  # Add corresponding timestamps
                "ID": IBM.trajectory.values,
                "color": f"C{i}"  # Add color based on trajectory number
            }
            for i in range(len(IBM.trajectory.values))
        ]
        trajectory_identifiers_b = IBM.trajectory.values

        # Check to see if a ds is supplied, if not, then the IBM data will take its place
        if ds is None:
            trajectories = trajectories_b
            trajectory_identifiers = trajectory_identifiers_b

    if IBM_2 is not None:
        trajectories_c = [
            {
                "lat_lon_pairs": np.column_stack((
                    (IBM_2.isel(trajectory=i).lat.values,
                     IBM_2.isel(trajectory=i).lon.values))
                ),
                "timestamps": IBM_2.isel(trajectory=i).date_seconds.values,  # Add corresponding timestamps
                "ID": IBM_2.trajectory.values,
                "color": f"C{i}"  # Add color based on trajectory number
            }
            for i in range(len(IBM_2.trajectory.values))
        ]
        trajectory_identifiers_c = IBM_2.trajectory.values

        # Check to see if other datasets are provided, if not, shuffle variables around to allow func to run
        if (ds is None) and (IBM is None):
            trajectories = trajectories_c
            trajectory_identifiers = trajectory_identifiers_c
        elif (ds is None) and (IBM is not None):
            trajectories_b = trajectories_c
            trajectory_identifiers_b = trajectory_identifiers_c 

    # Set up the figure
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.02, axes_class=plt.Axes)
    ax.set_extent(domain)
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    # If duration isn't supplied, set the number of frames to the length of the maximum trajectory
    if duration is not None:
        number_frames = range(0, (duration * int(24/ts)), plot_int) # Convert number of days into number of hours based on timestep
    else:
        number_frames = range(0, max(len(traj_data["lat_lon_pairs"]) for traj_data in trajectories), plot_int)

    # Loop through frames and save each at the specified interval
    for i in number_frames:
    #for i in range(max(len(traj_data["lat_lon_pairs"]) for traj_data in trajectories + trajectories_b)):
        ax.clear()
        ax.set_extent(domain)
        ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='gray')
        
        for traj_data in trajectories:
            lat_lon_pairs = traj_data["lat_lon_pairs"][:i+1]
            current_lat_lon = lat_lon_pairs[-1]  # Get current position
            
            # Plot previous locations with dashed line
            if len(lat_lon_pairs) > 1:
                if ds is not None: #Conditional formating to make the colours work between IBM and SVP data
                    ax.plot([lon for lat, lon in lat_lon_pairs[:-1]], [lat for lat, lon in lat_lon_pairs[:-1]],
                            color=traj_data["color"][1], alpha=0.8, transform=ccrs.PlateCarree())
                else:
                    ax.plot([lon for lat, lon in lat_lon_pairs[:-1]], [lat for lat, lon in lat_lon_pairs[:-1]],
                    color=traj_data["color"], alpha=0.8, transform=ccrs.PlateCarree())
            
            # Plot current position as shaped scatter point
            if ds is not None:
                ax.scatter(current_lat_lon[1], current_lat_lon[0], marker='o', zorder=2.7,
                           color=traj_data["color"][1], s=50, transform=ccrs.PlateCarree())
            else:
                ax.scatter(current_lat_lon[1], current_lat_lon[0], marker='o', zorder=2.6,
                           color=traj_data["color"], s=50, transform=ccrs.PlateCarree())

        if IBM is not None:
            for traj_data in trajectories_b:
                lat_lon_pairs = traj_data["lat_lon_pairs"][:i+1]
                current_lat_lon = lat_lon_pairs[-1]  # Get current position
                
                # Plot previous locations with dashed line
                if len(lat_lon_pairs) > 1:
                    ax.plot([lon for lat, lon in lat_lon_pairs[:-1]], [lat for lat, lon in lat_lon_pairs[:-1]],
                            color=traj_data["color"], linestyle='dotted', alpha=0.8, transform=ccrs.PlateCarree())
                
                # Plot current position as diamond-shaped scatter point
                ax.scatter(current_lat_lon[1], current_lat_lon[0], marker='D', color=traj_data["color"], s=45, transform=ccrs.PlateCarree())

        if IBM_2 is not None:
            for traj_data in trajectories_c:
                lat_lon_pairs = traj_data["lat_lon_pairs"][:i+1]
                current_lat_lon = lat_lon_pairs[-1]  # Get current position
                
                ## Plot previous locations with dashed line
                if len(lat_lon_pairs) > 1:
                    ax.plot([lon for lat, lon in lat_lon_pairs[:-1]], [lat for lat, lon in lat_lon_pairs[:-1]],
                            color=traj_data["color"], linestyle=(0, (1, 10)), alpha=0.8, transform=ccrs.PlateCarree())
                
                # Plot current position as triangle-shaped scatter point
                ax.scatter(current_lat_lon[1], current_lat_lon[0], marker='^', color=traj_data["color"],
                           s=45, transform=ccrs.PlateCarree(), zorder=2.5,)

        # Show timestamp at the specified interval
        #date_str = np.datetime_as_string(trajectories[0]["timestamps"][i],
        #                                 unit='m')  # Convert numpy datetime to string
        seconds = float(trajectories[0]["timestamps"][i])
        time_delta = timedelta(seconds=seconds)
        days = time_delta.days
        seconds_remaining = time_delta.seconds
        hours, remainder = divmod(seconds_remaining, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_format = f'{days} days, {hours}:{minutes}:{seconds}'
        ax.text(0.02, 0.95, time_format, transform=ax.transAxes, fontsize=12, fontweight="bold")

        # Show axis ticks
        ax.set_xticks(np.arange(domain[0], domain[1], 5))
        ax.set_yticks(np.arange(domain[2], domain[3], 5))

        # Create discrete color map for color bar if not supplied
        if (ds is not None) and (traj_cmap is None):
            traj_cmap, traj_norm = traj_cmap(ds)
        elif (ds is None) and (traj_cmap is None):
            traj_cmap = plt.cm.get_cmap("tab10", len(trajectory_identifiers))
            traj_norm = colors.BoundaryNorm(range(len(trajectory_identifiers) + 1), traj_cmap.N)
        
        # Create color bar using the list of identifiers
        if i == 0:
            cbar = plt.colorbar(mappable=plt.cm.ScalarMappable(cmap=traj_cmap, norm=traj_norm), cax=cax,
                                orientation="vertical", ticks=np.arange(len(trajectory_identifiers)) + 0.5)
            cbar.set_label("ID")
        
            cbar.ax.set_yticklabels(trajectory_identifiers)  # Set tick labels to IDs       

        # Create a legend
        handles = [
            plt.Line2D([], [], color='black', marker='o', linestyle='-', markersize=8),
            plt.Line2D([], [], color='black', marker='D', linestyle='--', markersize=8),
            plt.Line2D([], [], color='black', marker='^', linestyle='-.', markersize=8)
        ]
        legend = ax.legend(handles=handles, labels=labs, loc='center left', bbox_to_anchor=(0.0, 0.65),
                            frameon=False, fontsize=10, handlelength=2)
        ax.add_artist(legend)  # Add legend to the plot 

        plt.savefig(f"{framedir}/frame_{i:05d}.png")
        print(f"Saved frame {i}")

    print("Frames generated successfully.")


def drift_animation(framedir: str = 'C:/Users/z5493451/OneDrive - UNSW/Documents/Data/Drifters/animations',
                    outputdir: str = 'C:/Users/z5493451/OneDrive - UNSW/Documents/Data/Drifters/animations_full/temp.gif'):
    '''Converts a set of images into a GIF animation. Made to work alongside the drift_frames function.'''
    
    # Example frame generation #
    #fig, ax = plt.subplots()
    #for i in range(start_frame, end_frame, framerate):
    #    ax.clear()
    #    ax.plot(x[i], y[i]) #Plot data at each desired frame
    #    plt.savefig(framedir+f'/frame_{i:05d}.png') #Save the frame to a specified directory

    # Make the animation #
    frame_files = sorted(glob.glob(f'{framedir}/frame_*.png')) #Load the frames from a directory in order from frame_0...frame_n
    images = [Image.open(filename) for filename in frame_files] #Open frames with Image

    images[0].save(outputdir, save_all=True, append_images=images[1:], duration=100, loop=0) #Save the animation 
    print("Animation created from frames.")
