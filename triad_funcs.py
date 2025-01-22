# Drifter triad functions #

# lIBRARY #
# -----------------------------------------------------------------------

# Data manipulation #
import numpy as np
import xarray as xr
import pandas as pd
from collections import defaultdict

# Others #
from datetime import datetime
import os
from tqdm import tqdm

# visualization #
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import cmocean
from matplotlib.animation import FuncAnimation
from PIL import Image

# for url encoding and access #
from urllib.parse import quote
from urllib.parse import urlparse

# for distance and dispersion calcs
from geopy import distance

# Other Lagrangian analysis functions
from ldrift import drift_funcs as ld

def find_active_triads(ds, start_indices, num_arrays):

    pair_matches = defaultdict(lambda: defaultdict(set))
    
    # First, find all pairs and their shared times
    for i in range(num_arrays - 1):
        for j in range(i + 1, num_arrays):
            start_idx1, start_idx2 = start_indices[i], start_indices[j]
            end_idx1 = start_indices[i + 1] if i < num_arrays - 1 else len(ds.time.values)
            end_idx2 = start_indices[j + 1] if j < num_arrays - 1 else len(ds.time.values)
            
            time1 = set(ds.time[start_idx1:end_idx1].values)
            time2 = set(ds.time[start_idx2:end_idx2].values)
            
            shared_times = time1 & time2
            if shared_times:  # If there's an intersection
                id1, id2 = ds.ID[i].values, ds.ID[j].values
                pair_matches[id1][id2] = shared_times
                pair_matches[id2][id1] = shared_times
    
    # Now find triplets
    triplets = []
    for id1, matches in pair_matches.items():
        if len(matches) < 2:
            continue
        for id2 in matches:
            for id3 in matches:
                if id2 < id3 and id3 in pair_matches[id2]:
                    # Find the intersection of shared times among all three IDs
                    common_times = matches[id2] & matches[id3] & pair_matches[id2][id3]
                    if common_times:
                        # Add all common times for this triplet
                        for time in common_times:
                            triplets.append((id1, id2, id3, time))
    
    # Convert the result to a pandas DataFrame
    result_df = pd.DataFrame(triplets, columns=['ID1', 'ID2', 'ID3', 'SharedTime'])
    
    # Sort the DataFrame by IDs and then by SharedTime
    result_df = result_df.sort_values(['ID1', 'ID2', 'ID3', 'SharedTime'])
    
    return result_df, triplets


def chance_triads(pair_df, d0: any = 2500, t0: any = 6):
    """
    Find all triplets of drifters that were within d0 m of each other at some point in time.

    Parameters
    ----------
    pair_df : pd.DataFrame
        A pandas DataFrame containing columns 'ID_first', 'ID_second', 'time_first' and 'time_second'.
        Each row represents a pair of drifters that were within d0 km of each other at 'time_first' ~ 'time_second'.
        NOTE: A pair_df can be generated from a drifter ds using the ldrift 'find_chance_pairs' function.
    d0 : any, optional
        The distance (in m) that drifters must be within to be considered a pair. Defaults to 2500.
    t0 : any, optional
        The time (in hours) that drifters must be within to be considered a pair. Defaults to 6.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing columns 'ID1', 'ID2', 'ID3' and 'Shared_Time'. Each row represents a triplet of 
        drifters that were within d0 km of each other at roughly the same time.
    """
    # Create a dictionary to store all pairs for each ID
    id_pairs = defaultdict(set)
    
    # Create a dictionary to store shared times for each pair
    pair_times = defaultdict(set)
    
    # Populate the dictionaries
    for _, row in pair_df.iterrows():
        id1, id2 = row['ID'], row['comp_ID']
        time = row['time_first']  # We assume time_first and time_second are the same
        
        id_pairs[id1].add(id2)
        id_pairs[id2].add(id1)
        
        pair = tuple(sorted([id1, id2]))
        pair_times[pair].add(time)
    
    # Set to store unique triads
    unique_triads = set()
    # Find triplets
    triplets = []
    for id1 in tqdm(id_pairs):
        if len(id_pairs[id1]) < 2:
            continue
        for id2 in id_pairs[id1]:
            for id3 in id_pairs[id1]:
                if id2 < id3 and id3 in id_pairs[id2]:
                    # Create a sorted tuple of the triad for uniqueness check
                    triad = tuple(sorted([id1, id2, id3]))
                    
                    # Check if this triad has already been added
                    if triad in unique_triads:
                        continue

                    # Find shared times among all three pairs
                    times1 = pair_times[tuple(sorted([id1, id2]))]
                    times2 = pair_times[tuple(sorted([id1, id3]))]
                    times3 = pair_times[tuple(sorted([id2, id3]))]
                    
                    shared_times = times1 & times2 & times3 # Can use a time buffer value (t0) but not implementing here to keep accurate
                    
                    for time in shared_times:
                        triplets.append((id1, id2, id3, time))

                    unique_triads.add(triad)
    
    # Create result dataframe
    result_df = pd.DataFrame(triplets, columns=['ID1', 'ID2', 'ID3', 'Shared_Time'])
    
    # Sort the dataframe
    result_df = result_df.sort_values(['ID1', 'ID2', 'ID3', 'Shared_Time'])
    
    return result_df


def triad_datasets(ds: xr.Dataset, chance_triads_df: pd.DataFrame, max_delta=30):
    """
    Takes a pandas DataFrame containing a list of drifters that formed a triangle
    at some point in time and returns a dictionary containing the corresponding
    dataset for each triangle.

    Parameters
    ----------
    chance_triads_df : pd.DataFrame
        A pandas DataFrame containing a list of drifter IDs ('ID1', 'ID2', 'ID3') that formed a triangle at
        times 'Shared_Time'.
    ds: dataset containing drifter trajectories with the respective IDs in 'chance_triads_df'
    max_delta : number of days for the maximum trajectory length

    Returns
    -------
    triad_datasets : dict
        A dictionary containing the corresponding dataset for each triangle.
    """
    triad_datasets_dict = {'Triad': [], 'Dataset': []}
    ID_markers = chance_triads_df['ID1'].astype(str) + '_' + chance_triads_df['ID2'].astype(str) + '_' + chance_triads_df['ID3'].astype(str)
    unique_ids, unique_idx = np.unique(ID_markers, return_index=True)
    #print(unique_ids)
    #print('----------------------------------------')
    #print(unique_idx)
    ids1 = chance_triads_df['ID1'][unique_idx].astype(float).astype(np.int64()).values
    ids2 = chance_triads_df['ID2'][unique_idx].astype(float).astype(np.int64()).values
    ids3 = chance_triads_df['ID3'][unique_idx].astype(float).astype(np.int64()).values
    times = chance_triads_df['Shared_Time'][unique_idx].values.astype(np.datetime64)

    for i in range(len(unique_ids)):
        triad_ds = ld.retrieve_region(ds=ds, ids=[ids1[i], ids2[i], ids3[i]], min_time=times[i], full=False)
        print(triad_ds.ID.values)
        print(triad_ds.dt.values)
        #if triad_ds is not None:
        #    triad_ds = ld.retrieve_region(ds=triad_ds, dt_range=[0, max_delta*24*3600], full=False) #Pass through again to use the new dts
        triad_datasets_dict['Dataset'].append(triad_ds)
        triad_datasets_dict['Triad'].append(i)
    return triad_datasets_dict


def triangle_pars(lat1, lon1, lat2, lon2, lat3, lon3):
    # Check if all drifters are active
    # if any drifter isn't active, return NA
    if np.isnan(lat1) or np.isnan(lon1) or np.isnan(lat2) or np.isnan(lon2) or np.isnan(lat3) or np.isnan(lon3):
        triangle_area = np.nan
        lambda_ratio = np.nan
        max_angle = np.nan
        f = np.nan
    else:
        # Calculate side lengths using semi-spherical geometry
        coordinates = np.array([
            [lat1, lon1, lat2, lon2],
            [lat1, lon1, lat3, lon3],
            [lat2, lon2, lat3, lon3]
        ])

        # Calculate distances for all sides using vectorized operations
        distances = np.apply_along_axis(lambda x: distance.geodesic((x[0], x[1]), (x[2], x[3])).km, axis=1, arr=coordinates)

        # Assign the distances to variables a, b, and c
        a, b, c = distances
        # Calculate perimeter and semiperimeter
        p = a + b + c
        s = p / 2
        # Calculate area
        triangle_area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        # Calculate lambda ratio
        lambda_ratio = 12 * np.sqrt(3) * (triangle_area / p**2)
        # Calculate angles and find the max 
        angles = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)), np.arccos((c**2 + a**2 - b**2) / (2 * c * a)), np.arccos((a**2 + b**2 - c**2) / (2 * a * b))
        max_angle = np.degrees(np.max(angles))

        # Calculate the center latitude for Coriolis parameter
        center_lat = ((lat1 + lat2 + lat3) / 3)
        center_lon = ((lon1 + lon2 + lon3) / 3)

        # Earth's angular velocity (rad/s)
        omega = 7.2921159e-5

        # Coriolis factor
        f = 2 * omega * np.sin(np.radians(center_lat)) * 3600 # Raidians per hour
        
    return triangle_area, lambda_ratio, max_angle, f, center_lat, center_lon


def trig_pars_from_ds(triad_ds):

    idx, n = ld.get_traj_index(ds=triad_ds)
    min_array = min(len(triad_ds.time[slice(idx[0], n[0])].values),
                    len(triad_ds.time[slice(idx[1], n[1])].values),
                    len(triad_ds.time[slice(idx[2], n[2])].values))

    lat1, lon1 = triad_ds.latitude[slice(idx[0], n[0])].values[:min_array], triad_ds.longitude[slice(idx[0], n[0])].values[:min_array]
    lat2, lon2 = triad_ds.latitude[slice(idx[1], n[1])].values[:min_array], triad_ds.longitude[slice(idx[1], n[1])].values[:min_array]
    lat3, lon3 = triad_ds.latitude[slice(idx[2], n[2])].values[:min_array], triad_ds.longitude[slice(idx[2], n[2])].values[:min_array]
    time = triad_ds.time[slice(idx[0], n[0])].values[:min_array]
    dt = triad_ds.dt[slice(idx[0], n[0])].values[:min_array]

    areas, ratios, angles, fs, latbars, lonbars = [], [], [], [], [], []
    for i in range(min_array):
        area, ratio, angle, f, latbar, lonbar = triangle_pars(lat1[i], lon1[i], lat2[i], lon2[i], lat3[i], lon3[i])
        #areas.append(area / 1000**2) # / 1000**2 to convert m^2 to km^2
        areas.append(area)
        ratios.append(ratio)
        angles.append(angle)
        fs.append(f)
        latbars.append(latbar)
        lonbars.append(lonbar)
    
    # Calculate divergence
    A_inv = 1 / np.array(areas)
    dA = np.diff(areas)
    diff_dt = np.diff(dt / 3600)
    dAdt = np.append(0, dA / diff_dt)
    D = A_inv * dAdt
    
    return pd.DataFrame({'time': time, 'dt': dt, 'area': areas,
    'lambda_ratio': ratios, 'max_angle': angles, 'divergence': D,
    'f': fs, 'center_lat': latbars, 'center_lon': lonbars})


def lambda_qc(df, columns_to_filter=None, threshold=0.2):
    '''
    Filters out specific values from specified columns in a dataframe where the corresponding
    lambda ratio exceeds a threshold value.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing 'lambda_ratio' and columns to be filtered
    columns_to_filter : list or None
        List of column names to apply the filtering. If None, applies to all columns except 'lambda_ratio'
    threshold : float, default=0.2
        Threshold value for lambda ratio filtering
        
    Returns:
    --------
    pandas.DataFrame
        Copy of input dataframe with values set to nan where lambda ratio exceeds threshold
    '''
    # Create a copy of the input dataframe
    new_df = df.copy()
    
    # If no columns specified, filter all columns except lambda_ratio
    if columns_to_filter is None:
        columns_to_filter = [col for col in df.columns if col != 'lambda_ratio']
    
    # Create boolean mask for indices where lambda_ratio exceeds threshold
    mask = new_df['lambda_ratio'] < threshold
    
    # Apply nan only to specified columns at masked indices
    for column in columns_to_filter:
        new_df.loc[mask, column] = np.nan
    
    return new_df


def plot_triangles(lat_lon_arrays, domain: list = [145, 175, -45, -15], borders: bool = True,
                   cvec: any = 'blue', bathy: any = None):
    '''
    Function to plot triangles formed by drifters onto a map
    
    Parameters:
    -----------
    lat_lon_arrays : list of lists
        List of lists containing the latitude and longitude coordinates of the drifters
    domain : list, optional
        List of four values defining the extent of the map [min lon, max lon, min lat, max lat]
    borders : bool, optional
        If True, adds the coastline and land features to the map
    cvec : any, optional
        Colours to be used for the triangles
    bathy : any, optional
        Bathymetry data to be used for the contours
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the plot
    
    Example:
    --------
    >>> from ldrift import triad_funcs
    >>> lat_lon_arrays = [
            [-30.1, -30.2, -30.3, -30.4],
            [151.5, 151.6, 151.7, 151.5],    ...     
            [-31.1, -31.2, -31.3, -31.4],
            [152.0, 152.1, 152.2, 152.0],   ...     
            [-32.1, -32.2, -32.3, -32.4],
            [152.3, 152.4, 152.5, 152.3]
    ... ]
    >>> fig = triad_funcs.plot_triangles(lat_lon_arrays, domain=[151, 153, -33, -32])
    >>> fig.show()
    '''
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Set the map extent based on the specified region
    ax.set_extent(domain, crs=ccrs.PlateCarree())

    # Add borders and features
    if borders:
        ax.add_feature(cfeature.LAND, facecolor='grey', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.25, zorder=1)

    if bathy is not None:
        latBathy = bathy.variables['lat'][:]       
        lonBathy = bathy.variables['lon'][:]       
        heightBathy = bathy.variables['height'][:]
        ax.contour(lonBathy, latBathy, heightBathy,
                    levels = [-2000, -1000, -200], linewidths=1,
                    colors = 'grey', linestyles='solid', zorder=0.2)
    
    # Create colour map and apply to 'cvec'
    cmap = cmocean.cm.speed
    norm = colors.Normalize(vmin=-1, vmax=1)

    alpha = 0.65  # Initial alpha value
    for i in range(0, len(lat_lon_arrays[0]), 3):
        try:
            lats = [lat_lon_array[i] for lat_lon_array in lat_lon_arrays[::2]]
            lons = [lat_lon_array[i] for lat_lon_array in lat_lon_arrays[1::2]]
        except IndexError as e:
            print('IndexError at i = ', i, e)
            break

        for j in range(len(lats)):
            triangle = plt.Polygon([(lons[j], lats[j]),
                                    (lons[(j + 1) % len(lats)], lats[(j + 1) % len(lats)]),
                                    (lons[(j + 2) % len(lats)], lats[(j + 2) % len(lats)])],
                                    edgecolor='black', facecolor=cmap(norm(cvec[j])), transform=ccrs.PlateCarree(),
                                    alpha=alpha
                                    )
            ax.add_patch(triangle)

            #alpha -= 0.1  # Reduce alpha for the next triangle

    # Add coloUr bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Normalized Divergence')

    # Set the displayed region
    if domain:
        ax.set_extent(domain, crs=ccrs.PlateCarree())
        ax.set_xticks(np.arange(domain[0],domain[1],5), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(domain[2],domain[3],5), crs=ccrs.PlateCarree())

    ax.set_title('Drifter Triad Time Series')
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    return fig



def animate_triangles(lat_lon_arrays, domain: list = [145, 175, -45, -15],
                      borders: bool = True, save_path: str = any, cvec: any = 'blue',
                      bathy: any = None, framerate = 30):
    ''' Function to create an animation of triangles formed by drifters over time'''

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Set the map extent based on the specified region
    ax.set_extent(domain, crs=ccrs.PlateCarree())

    # Add borders and features
    if borders:
        ax.add_feature(cfeature.LAND, facecolor='grey', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.25, zorder=1)

    if bathy is not None:
        latBathy = bathy.variables['lat'][:]       
        lonBathy = bathy.variables['lon'][:]       
        heightBathy = bathy.variables['height'][:]
        ax.contour(lonBathy, latBathy, heightBathy,
                    levels = [-2000, -1000, -200], linewidths=1,
                    colors = 'grey', linestyles='solid', zorder=0.2)

    alpha = 0.65  # Initial alpha value

    def update(frame):
        #ax.clear()  # Clear the previous frame

        lats = [lat_lon_array[frame] for lat_lon_array in lat_lon_arrays[::2]]
        lons = [lat_lon_array[frame] for lat_lon_array in lat_lon_arrays[1::2]]

        # Create colour map and apply to 'cvec'
        cmap = cmocean.cm.speed
        norm = colors.Normalize(vmin=-3, vmax=3)
        for j in range(len(lats)):
            triangle = plt.Polygon([(lons[j], lats[j]),
                                    (lons[(j + 1) % len(lats)], lats[(j + 1) % len(lats)]),
                                    (lons[(j + 2) % len(lats)], lats[(j + 2) % len(lats)])],
                                    edgecolor='black', facecolor=cmap(norm(cvec[j])), transform=ccrs.PlateCarree(),
                                    alpha=alpha
                                    )
            ax.add_patch(triangle)

        # Set the displayed region
        if domain:
            ax.set_extent(domain, crs=ccrs.PlateCarree())
            ax.set_xticks(np.arange(domain[0],domain[1],5), crs=ccrs.PlateCarree())
            ax.set_yticks(np.arange(domain[2],domain[3],5), crs=ccrs.PlateCarree())

        ax.set_title(f'Drifter Triad at timestep {frame}')
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())

    anim = FuncAnimation(fig, update, frames=len(lat_lon_arrays[0]), repeat=False, blit=False)

    # Save the animation as a GIF
    anim.save(save_path, writer='pillow', fps=framerate)


