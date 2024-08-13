################################
## SVP and Opendrift Functions #
################################
''' This script contains all the functions I use to analyse the NOAA GDP dataset and 
    some of the functions to analyse partilce tracking models (IBM; e.g., OpenDrift)
    output (some of the GDP functions are fairly robust and can be run with IBM
    input regardless)
 '''

## LIBRARY - included for convenience and for documentation to create an environment file ## ===========

# Data manipulation
import numpy as np

import xarray as xr
import pandas as pd

#from numba import njit

# Analysis
from scipy import stats
from scipy.stats import linregress
from scipy.stats import kstest, norm, expon, powerlaw, skew, kurtosis
#from scipy.optimize import curve_fit
#from scipy.optimize import minimize
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter # For diffusivity tensors
#from powerlaw import Fit
from pwlf import PiecewiseLinFit
from sklearn.decomposition import PCA # PCA for generating diffusivity tensors
from sklearn.impute import SimpleImputer # For dealing with NaNs during PCA

# GIS and such
from geopy import distance
import geopy as gp
import geopandas as gpd
import pyproj
from shapely.geometry import Point
from shapely.geometry import box
from shapely.geometry import Polygon

# Others
from itertools import combinations
from tqdm import tqdm
from collections import Counter
import time
import math
import os
import datetime as datetime
import glob
import warnings

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import seaborn as sns
import cmocean
import colormaps as cmaps
import matplotlib.animation as animation # For animations
from PIL import Image # For animations

# For ragged array construction
from ldrift.preproc import create_ragged_arr

## FUN-CTIONS - so called because they are Fun, not Necessarily "functional" ## =====================

def get_rowsize(ds: any = None, array: any = None) -> np.array:
    ''' Function to calculate the 'rowsize' variable in a dataset such as the GDPv2.0 file.
        It is mainly used when subseting so that each unique trajectory is preserved and 
        can thus be accessed using chunks or blocks of sizes equal to the rowsize.
        Takes a dataset or an array of ids as input'''
    
    if array is not None: # Set the array to ds ids otherwise just use the array of supplied ids
        pass
    elif ds is not None:
        array = ds.ids.values

    # Set rowsizes to match the number of entries for each unique ID
    rs_temp = Counter(array) 
    new_rs = np.full(len(rs_temp),fill_value=0)
    for i, value in enumerate(rs_temp):
        new_rs[i] = rs_temp.get(value, 0)

    return new_rs


def get_traj_index(ds) -> np.array:
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


def get_dt(ds, ts: int = 3600) -> xr.Dataset:
    ''' Function to add a delta-time (time since deployment) column to a drifter dataset
    
    Args:
        ds: dataset of drifter trajectories
        ts: timestep (default 3600 s) between each observation
    Returns:
        A copy of the original dataset with a new dt variable'''

    traj_idx, trajsum = get_traj_index(ds)

    dt_arr = []
    for i in range(len(ds.traj)):
        dt = (ds.time[traj_idx[i]:trajsum[i]].values - ds.time[traj_idx[i]].values).astype('timedelta64[s]').view('int64')
        dt_arr.append(dt)

    #if (len(dt_arr) > 0):
    #    dt_arr = np.concatenate(dt_arr)
    dt_arr = np.concatenate(dt_arr)

    # Set the dt variable in the dataset
    dt = xr.DataArray(dt_arr, coords={'ids': ds.ids}, dims=['obs']) 

    return ds.assign(dt=dt) # For dt using the gap variable, see 'Lukes old drifter codes'


def find_index(arr, threshold, x):
    """
    Find the index of the first occurrence of a target value in an array, given a threshold.

    Args:
        arr (numpy.ndarray): The input array.
        threshold (int or float): The threshold value.
        x (int or float): The target value.

    Returns:
        int or float: The index of the first occurrence of the target value, or NaN if not found.
    """
    target_value = x - threshold # i.e., t_n + tau (minus because we use flipped values)
    mask = arr == target_value  # Create a boolean mask for the condition
    if np.any(mask):
        return np.argmax(mask)  # Return the index of the first occurrence
    return np.nan  # Return np.nan if no match is found


def traj_cmap(traj, random_seed: int = 42) -> tuple[colors.ListedColormap, colors.BoundaryNorm]:
    '''
    Function to create a colormap based on the length of a dataset.

    Args:
        traj: array of trajectory IDs
        random_seed: integer seed for consistent color generation

    Returns:
        A tuple containing a ListedColormap and a BoundaryNorm object
    '''

    # Find unique IDs just in case there are duplicates
    unique_traj_ids = np.unique(traj)
    num_unique_trajectories = len(unique_traj_ids)
        
    # Generate a consistent set of colors based on trajectory IDs
    np.random.seed(random_seed) # Seed number is what makes it consistent
    random_colors = np.random.rand(num_unique_trajectories, 3)
        
    # Create a colormap using the consistent colors
    cmap = colors.ListedColormap(random_colors) # Create a colormap object
    norm = colors.BoundaryNorm(range(len(traj) + 1), cmap.N) # Create a BoundaryNorm object
    
    return cmap, norm # Return the colormap and normalizer object


def get_traj_cols(ds: any = None, df: any = None) -> xr.Dataset:
    ''' Short function used to set a colour ID (from 0 to length of traj dimension) for each
        unique drifter in the ds for plotting. Plan to add in opacity options in the future.'''
    
    if df is not None:
        # Create a colormap using the consistent colors
        trajs = pd.factorize(df['ID_marker'].unique())[0] # List of unique trajectory #s from 1,...,n
        cmap, _ = traj_cmap(trajs)

        # Create an array of trajectory ids based on 'rowsize'
        traj = pd.factorize(df['ID_marker'].values)[0] # Trajectory numbers for entire df 1,1,1,...,n,n,n

        # Create a list of color values corresponding to each trajectory #
        traj_cols = [cmap(i) for i in traj]

        # Convert colormap values to hexadecimal
        traj_cols_hex = [colors.to_hex(color) for color in traj_cols]

        # Set a variable in the dataframe to store color values unique to a trajectory
        df['traj_cols'] = pd.Series(traj_cols_hex, dtype=str, index=None)

        # Return the dataframe with the 'traj_cols' variable added
        return df
    
    else:
        # Create a colormap using the consistent colors
        cmap, _ = traj_cmap(ds.traj.values)

        # Create an array of trajectory ids based on 'rowsize'
        trajs = np.repeat(ds.traj.values, ds.rowsize.values)

        # Create a list of color values corresponding to each trajectory id
        traj_cols = [cmap(i) for i in trajs]

        # Convert colormap values to hexadecimal
        traj_cols_hex = [colors.to_hex(color) for color in traj_cols]

        # Set a variable in the dataset to store color values unique to a trajectory
        traj_cols_da = xr.DataArray(traj_cols_hex, coords={'ids': ds.ids}, dims=['obs'])

        # Return the dataset with the 'traj_cols' variable added
        return ds.assign(traj_cols=traj_cols_da)


def gap_check(ds):
    ''''''
    traj_i, trajsum = get_traj_index(ds)

    ids = ds.ID.values

    max_gaps = []
    for i in range(len(ds.traj)):
        max_gap = max(ds.gap.values[slice(traj_i[i],trajsum[i])])
        max_gaps.append(max_gap)
    max_gaps = np.concatenate(max_gaps)

    data = {'ID': ids, 'max_gap': max_gaps}
    return pd.DataFrame(data)


def drift_meta(ds, op_type: str = 'd'):
    ''' Module to collect 'start' times and longitudes
    from large drifter datasets. Returns a pandas dataframe
    which is formatted to be used with my opendrift code'''

    #Create a drifter index 
    traj_index, trajsum = get_traj_index(ds)
 
    #Find start times and locations
    start_lat = ds.latitude[traj_index].values
    start_lon = ds.longitude[traj_index].values
    start_time = ds.time[traj_index].values
    end_time = ds.time[trajsum-1].values
    #end_time = ds.end_date.values

    #Find Identifiers
    WMO = ds.WMO.values
    ID = ds.ID.values

    #Set columns and rows
    rows = range(0,len(traj_index))
    columns = ['ID','WMO','start_lat','start_lon','start_time','end_time']

    #Concatenate data and load into a pandas df
    data = np.transpose([ID,WMO,start_lat,start_lon,start_time.astype('datetime64[s]'),end_time.astype('datetime64[s]')])
    df = pd.DataFrame(data=data, index=rows, columns=columns)
    d = df.to_dict()

    # Return either the pd.dataframe or a dict
    if op_type == 'df':
        return df
    else:
        return d


def meta_series(ds: any = None, trajectory_data: any = None, max_days: int = 250,
                step_size: int = 30, omit: bool = True,
                tick_scaler: any = 10) -> plt.figure:
    ''' Plots a time series of the number of drifters active by year
    
    Args:
        ds: A drifter dataset - use if you want number of drifters for each labelled year
        trajectory_data: A dict of dt values, colours, and labels - use if you want a "survival plot" across an interval of days

    Example trajectory_data: trajectory_data = {
    'Trajectory 1': {'array': arr1, 'color': cmaps.WhBlReWh(44)},
    'Trajectory 2': {'array': arr2, 'color': cmaps.WhBlReWh(53)},
    'Trajectory 3': {'array': arr3}  # No color specified, will use default
    }
    '''

    fig, ax = plt.subplots(figsize=(4,3.5), dpi=125)

    if (ds is not None) and (trajectory_data is None):
        print('Whole ds supplied, plotting yearly time series')

        # Extract start times
        df = drift_meta(ds, op_type='df')

        # Create a list to store the yearly counts
        yearly_counts = []

        # Loop through each row in the DataFrame
        for _, row in df.iterrows():
            start_year = row['start_time'].year
            end_year = row['end_time'].year

            # Generate a list of years the data point is active
            years_active = list(range(int(start_year), int(end_year) + 1))

            # Add the years to the list
            yearly_counts.extend(years_active)

        # Create a Pandas Series to count the occurrences of each year
        yearly_counts = pd.Series(yearly_counts)

        # Count the occurrences of each year
        yearly_counts = yearly_counts.value_counts().sort_index()

        # Create a bar plot
        ax.bar(yearly_counts.index, yearly_counts, color='#002060')

        # Set labels and title
        ax.xlabel('Year')
        ax.ylabel('Number of drifters')
        ax.title('Time series of active drifters in the EAC by year')

    if (ds is None) and (trajectory_data is not None):
        print('Array supplied, plotting survival')
        for label, data in trajectory_data.items():
            dt = data['array']
            color = data.get('color', None)  # Use default color if not specified

            dt_days = dt / 3600 / 24  # Convert dt stored in seconds to days
            # Filter the time values to only include data up to max_days
            filtered_time_values = dt_days[dt_days <= max_days]
            # Calculate the unique time steps and their counts
            unique_times, counts = np.unique(filtered_time_values, return_counts=True)

            #percentile = 25  # n-th percentile
            #percentile_cutoff = percentile / 100 * total_trajs
            #percentile_time = np.max(unique_times[counts >= percentile_cutoff])

            print(label, 'total trajectories: ', data['total_traj'])
            #print(percentile_cutoff)
            #print(percentile, 'percent of drifters were active at:', percentile_time)

            # Select every nth unique time step
            #selected_times = unique_times[::step_size * 24]
            #selected_counts = counts[::step_size * 24]
            selected_times = unique_times
            selected_counts = counts

            # Step plot
            if omit:
                ax.step(selected_times[1:], selected_counts[1:], where='post', lw=1, label=label, color=color)
            else:
                ax.step(selected_times, selected_counts, where='post', lw=1, label=label, color=color)
            
            # Add label above the line
            ax.text(selected_times[6], selected_counts[1], label, ha='center', va='bottom', fontsize=10)

        plt.xscale('log')
        plt.xlabel('Time (Days)', fontsize=10)
        plt.ylabel('Number of Trajectories', fontsize=10)
        #plt.ylim(0, selected_counts[1] + tick_scaler)
        plt.ylim(0, tick_scaler)
        #plt.legend(fontsize=10)
        plt.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5)
        #major_xticks = np.arange(0, max_days + 1, (5 * step_size) * max_days / tick_scaler)
        #plt.xticks(major_xticks, fontsize=10)
        plt.yticks(fontsize=10)
        plt.show()

    return fig


def grid_data(points, u: any = None, v: any = None, total: any = None,
              sst: any = None, time: any = None,  displacement: any = None,
              disp_x: any = None, disp_y: any = None, d_prime: any = None,
              disp_x_prime: any = None, disp_y_prime: any = None,
              d_x2: any = None, d_y2: any = None, d_A2: any = None,
              total_prime: any = None, u_prime: any = None,
              v_prime: any = None, Davis_diff: any = None,
              res = 0.25, domain: list = [145,165,-45,-20],
              threshold = 30):
    ''' Grids the mean of a selected drifter data variable on a res x res degree grid
    
    args:
        points: tuple of lat/lon coordinates ([lat],[lon])
        grid vars (u, v, total, sst, time, disp, d_x2, d_y2, d_A2, u', v', tot'):arrays of variable data to be gridded
        res: resolution (in degrees) of grid
        domain: vertices defining the total area for the grid to cover
        threshold: the minimum no. datapoints in a cell - returns nan for that cell if less
        '''
    
    lon = np.arange(domain[0], domain[1], res)
    lat = np.arange(domain[2], domain[3], res)

    # Create a test df
    arrays = [u, v, total, sst, time, displacement, disp_x, disp_y,
              d_prime, disp_x_prime, disp_y_prime, d_x2, d_y2, d_A2, total_prime, u_prime, v_prime, Davis_diff]
    data = {name: array for name, array in zip(['u', 'v', 'total', 'sst', 'time', 'displacement', 'disp_x', 'disp_y',
                                                'd_prime', 'disp_x_prime', 'disp_y_prime', 'd_x2', 'd_y2', 'd_A2',
                                                'total_prime', 'u_prime', 'v_prime', 'Davis_diff'], arrays) if array is not None}
    test_df = pd.DataFrame(data)

    # Use binned_statistic_2d to calculate mean over regular grid
    result_dict = {}

    for var_name in test_df.columns:
        stats_result = stats.binned_statistic_2d(
            points[:, 0], points[:, 1], test_df[var_name], statistic=np.nanmean, bins=[lon, lat]
            )
        
        count_result = stats.binned_statistic_2d(
            points[:, 0], points[:, 1], test_df[var_name], statistic='count', bins=[lon, lat]
            )
        
        mean = stats_result.statistic.T.flatten()
        count = count_result.statistic.T.flatten()

        # Check if the count is greater than the specified threshold
        invalid_indices = np.where(count < threshold)
        mean[invalid_indices] = np.nan

        result_dict[var_name] = mean

        # Extract lat and lons once
        if 'lat' not in result_dict:
          lats = 0.5 * (stats_result.y_edge[:-1] + stats_result.y_edge[1:])
          lons = 0.5 * (stats_result.x_edge[:-1] + stats_result.x_edge[1:])
          lons_mesh, lats_mesh = np.meshgrid(lons, lats)
          result_dict['lat'] = lats_mesh.flatten()
          result_dict['lon'] = lons_mesh.flatten()

    # Create DataFrame
    op_df = pd.DataFrame(result_dict)

    return op_df


def plot_meshgrid_on_map(df, grid_var, domain = [145,165,-45,-20], sigma=1.5,
                         cb_range: any = [1000, 80000], unit_conv = 1000,
                         var_label = r'Diffusivity ($m^2 s^{-1}$)'):

    # Extract data from the DataFrame
    lon = df['lon'].values.reshape(len(np.unique(df['lat'])), -1)
    lat = df['lat'].values.reshape(len(np.unique(df['lat'])), -1)
    mean_value = abs(df[grid_var].values.reshape(len(np.unique(df['lat'])), -1)) * unit_conv

    ## Create a meshgrid for interpolation
    xi, yi = np.meshgrid(np.linspace(lon.min(), lon.max(), lon.shape[1]),
                         np.linspace(lat.min(), lat.max(), lat.shape[0]))
    
    # Mask land and nans
    mask = ~np.isnan(mean_value)

    # Interpolate missing values
    mean_value = griddata((lon[mask], lat[mask]), mean_value[mask], (xi, yi), method='linear')

    # Apply Gaussian filter to the interpolated mean values
    mean_value = gaussian_filter(mean_value, sigma=sigma, cval=np.nan)

    # Create a map with specified domain
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},figsize=(6,4), dpi=150)
    ax.set_extent([domain[0], domain[1], domain[2], domain[3]])

    # Plot meshgrid
    #pcm = ax.pcolormesh(lon, lat, mean_value,
    #                    cmap=cmaps.NCV_jet, shading='auto',
    #                    transform=ccrs.PlateCarree(), norm=colors.LogNorm())

    pcm = ax.pcolormesh(xi, yi, mean_value, cmap='cividis', shading='auto',
                        transform=ccrs.PlateCarree(), norm=colors.LogNorm(vmin=cb_range[0], vmax=cb_range[1]))

    # Plot contours
    #contour_levels = np.logspace(3, 5, 8)
    #contour_levels = np.arange(10**3, 10**4.8, 9)
    contour_levels = [1000, 5000, 10000, 15000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    cs = ax.contour(xi, yi, mean_value, levels=contour_levels, colors='k', linewidths=0.5, transform=ccrs.PlateCarree())

    # Label every second contour level
    #fmt = '%1.1e'
    ax.clabel(cs, cs.levels[::], inline=True, fontsize=6)

    # Plot mean quiver arrows on top
    #ax.quiver(lon, lat, df.u.values, df.v.values, scale=20, color='black', width=0.0025)

    ## Plot deviation quiver arrows on top
    #ax.quiver(lon, lat, df.u_prime.values, df.v_prime.values, scale=0.01, color='black', width=0.0025)

    # Add land feature
    ax.coastlines(linewidth=0.8, zorder=2.5)
    ax.add_feature(cfeature.LAND, edgecolor='black', color='gainsboro', zorder=2)

    # Add gridlines
    ax.gridlines(linewidth=0.5, linestyle='--', x_inline=False, y_inline=False)

    # Add colorbar
    cbar = fig.colorbar(pcm)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(0.5)

    # Customize colorbar ticks to show original values
    cbar.set_ticks([10**i for i in range(int(np.log10(cb_range[0])), int(np.log10(cb_range[1])) + 1)])
    cbar.set_ticklabels([f'{10**i}' for i in range(int(np.log10(cb_range[0])), int(np.log10(cb_range[1])) + 1)])
    cbar.set_label(var_label, fontsize=9)

    # Set labels and title
    ax.set_xticks(np.arange(domain[0],domain[1],5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(domain[2],domain[3],5), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    #ax.set_xlabel('Longitude')
    #ax.set_ylabel('Latitude')
    #plt.title('Mean Values on Map')
    return fig


def disp_tensor(ds, res: any = 0.25, domain = [145,165,-45,-20]):
    ''' Takes a drifter dataset that contains observations of displacement values
        and returns a new ds with the mean and deviation of these displacements
        in a specified grid resolution'''
    
    lon_lat_points = np.column_stack((ds.longitude.values, ds.latitude.values))
    total_velocity = np.sqrt(ds.ve.values ** 2 + ds.vn.values ** 2)
    grid_df = grid_data(points=lon_lat_points, displacement=ds.pt_d.values,
                                     disp_x=ds.pt_x.values, disp_y=ds.pt_y.values,
                                     u=ds.ve.values, v=ds.vn.values, total=total_velocity,
                                     res=res, domain=domain)
    
    # Finding where each lat/lon falls in the gridded data and use the indices to find mean vals
    gridded_tree = cKDTree(np.column_stack((grid_df['lon'], grid_df['lat']))) #KDTree for the gridded data
    _, indices = gridded_tree.query(lon_lat_points, k=1) #Find closest pts

    # Find mean vals to use in subtraction process
    closest_totals = grid_df['total'].iloc[indices].values #Corresponding values for closest pts
    closest_u = grid_df['u'].iloc[indices].values
    closest_v = grid_df['v'].iloc[indices].values
    closest_displacement = grid_df['displacement'].iloc[indices].values
    closest_disp_x = grid_df['disp_x'].iloc[indices].values
    closest_disp_y = grid_df['disp_y'].iloc[indices].values

    # Store time lags
    traj_idx, trajsum = get_traj_index(ds=ds)
    time_lags = ds.dt.values
    time_lags[traj_idx] = 3600

    #Subtract means from individual observations to get deviations
    dev_totals = closest_totals - total_velocity
    u_deviations = ds.ve.values - closest_u
    v_deviations = ds.vn.values - closest_v
    dev_displacement = abs(ds.pt_d.values - closest_displacement)
    x_deviations = abs(ds.pt_x.values - closest_disp_x)
    y_deviations = abs(ds.pt_y.values - closest_disp_y)

    # Perform PCA on deviation arrays to decompose diff tensor 
    ##### DISPERSION TENSOR 'S' ###############
    # Compute the single-particle dispersion tensor
    s_xx = (x_deviations**2)
    s_yy = (y_deviations**2)
    S = np.sqrt(s_xx + s_yy)
    #s_xy = np.mean(x_deviations * y_deviations, axis=0)

    dt = np.gradient(time_lags)
    dt[traj_idx] = 0
    ds_2 = np.gradient(S)
    ds_2[traj_idx] = np.NAN
    s_2 = 0.5 * (ds_2 / dt)

    ##### DAVIS TENSOR 'K' ###############
    # Compute the diffusivity tensor
    k_xx = (u_deviations * x_deviations)
    k_yy = (v_deviations * y_deviations)
    k_xy = (u_deviations * y_deviations)
    k_yx = (v_deviations * x_deviations)

    k_s_xx = k_xx
    k_s_yy = k_yy
    k_s_xy = (k_xy + k_yx) / 2
    k_s_yx = k_s_xy

    k_s_1, k_s_2 = np.linalg.eigh(np.array([[k_s_xx, k_s_xy], [k_s_yx, k_s_yy]]).transpose(2, 0, 1))
    #print(k_s_1[..., 0])

    # Average the minor components of kjk and sjk to calculate lateral diffusivity
    k_star_2 = s_2
    k_2 = k_s_1[..., 0]
    lateral_diffusivity = 0.5 * (k_2 + k_star_2)
    #lateral_diffusivity = k_s_1[..., 0]
    
    # Create new ds
    u_prime = xr.DataArray(u_deviations, coords={'ids': ds.ids}, dims=['obs'])
    v_prime = xr.DataArray(v_deviations, coords={'ids': ds.ids}, dims=['obs'])
    total_prime = xr.DataArray(dev_totals, coords={'ids': ds.ids}, dims=['obs'])
    disp_x_prime = xr.DataArray(x_deviations, coords={'ids': ds.ids}, dims=['obs'])
    disp_y_prime = xr.DataArray(y_deviations, coords={'ids': ds.ids}, dims=['obs'])
    d_prime = xr.DataArray(dev_displacement, coords={'ids': ds.ids}, dims=['obs'])
    lateral_diffusivity = xr.DataArray(lateral_diffusivity, coords={'ids': ds.ids}, dims=['obs'])


    new_ds = ds.assign(d_prime=d_prime, disp_x_prime=disp_x_prime,
                   disp_y_prime=disp_y_prime,
                   total_prime=total_prime, v_prime=v_prime,
                   u_prime=u_prime, lateral_diffusivity=lateral_diffusivity
                   )
    return new_ds


def grid_plot(ds, domain = [145,165,-45,-20], n = 0.25, threshold = [30, 3000000]) -> plt.figure:
    ''' Plots the number of drifters within bins on a map'''
 
    # Create a nxn degree grid
    lat_bins = np.arange(domain[2], domain[3], n)
    lon_bins = np.arange(domain[0], domain[1], n)

    # Compute the 2D histogram to get data density
    hist, xedges, yedges = np.histogram2d(
        ds.longitude, ds.latitude, bins=[lon_bins, lat_bins]
    )
    print(hist.max())

    # Apply threshold to filter values
    hist[(hist < threshold[0])] = np.nan

    # Create a figure and set the projection
    fig = plt.figure(figsize=(6,4), dpi=125)
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Plot the heatmap
    plt.pcolormesh(xedges, yedges, (hist.T), zorder = 0.5,
                   cmap='cividis', norm=colors.LogNorm(vmax=threshold[1]))

    # Add coastlines and gridlines
    ax.gridlines(linestyle='--', linewidth=0.5, zorder = 0.7)
    ax.coastlines(linewidth=0.8)
    ax.add_feature(cfeature.LAND, edgecolor='black', color='gainsboro', zorder = 0.6)
    #ax.add_feature(cfeature.LAND, edgecolor='black', color='gainsboro', zorder = 0.8)
    ax.set_extent(domain, crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(domain[0],domain[1],5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(domain[2],domain[3],5), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(axis='both', which='major', labelsize=9)

    # Add in bounding boxes
    ax.plot([150, 155, 155, 150, 150],
        [-33, -33, -25, -25, -33],
        c='black', transform=ccrs.PlateCarree(), lw=1.2, linestyle='-', zorder = 0.75)
    ax.plot([150, 155, 155, 150, 150],
        [-40, -40, -33, -33, -40],
        c='black', transform=ccrs.PlateCarree(), lw=1.2, linestyle='-', zorder = 0.75)
    
    # Show the plot
    cbar = plt.colorbar(ax.collections[0], ax=ax, label='Number of observations')
    cbar.set_label('Number of observations', fontsize=9)
    cbar.outline.set_linewidth(0.5)

    # Customize colorbar ticks to show original values
    cbar.set_ticks([10**i for i in range(int(np.log10(threshold[0])), int(np.log10(threshold[1])) + 1)])
    cbar.set_ticklabels([f'{10**i}' for i in range(int(np.log10(threshold[0])), int(np.log10(threshold[1])) + 1)])
    cbar.ax.tick_params(labelsize=9)
    plt.show()

    return fig


def PDF_plot(A2, type='norm', n=30) -> plt.figure:
    ''' Plots the probability/cumulative distribution function of an array of data.
        By default, a KS test is also performed between the generated PDF and a
        Gaussian with the same mean and std deviation.'''
    
    # Use min and max of the given data
    min_val = A2.min()
    max_val = A2.max()

    # Create a histogram and PDF
    fig, ax = plt.subplots(figsize=(8, 8))

    if type == 'norm':
        # Calculate the Gaussian PDF with parameters from A2
        mu = A2.mean()
        sigma = A2.std()
        x = np.linspace(min_val, max_val, 100)
        pdf = norm.pdf(x, loc=mu, scale=sigma)

        # Plot the normal dist
        ax.plot(x, pdf, 'r-', label='Gaussian')

        # Perform KS test
        statistic, p_value = kstest(A2, "norm", args=(mu, sigma))
        print("Gaussian KS Statistic:", statistic)
        print("Gaussian p-value:", p_value)

    else:
        # Calculate and plot the exponential PDF
        scale = A2.mean()
        x_exp = np.linspace(0, max_val, 100)
        pdf_exp = expon.pdf(x_exp, scale=scale)
        ax.plot(x_exp, pdf_exp, 'b-', label='Exponential')

        # Perform KS test for exponential
        statistic_exp, p_value_exp = kstest(A2, "expon", args=(0, scale))
        print("Exponential KS Statistic:", statistic_exp)
        print("Exponential p-value:", p_value_exp)

        # Calculate and plot the power law PDF
        a = 4/3
        x_power = np.linspace(0.1, max_val, 100)  # Avoid division by zero
        pdf_power = (a * x_power ** (a - 1)) / (max_val ** a)
        ax.plot(x_power, pdf_power, 'g-', label='Power Law (x^(4/3))')

        # Perform KS test for power law
        statistic_power, p_value_power = kstest(A2, lambda x: powerlaw.cdf(x, a))
        print("Power Law KS Statistic:", statistic_power)
        print("Power Law p-value:", p_value_power)

    # Calculate histogram data as bars
    counts, bins = np.histogram(A2, bins=n, density=True)
    bin_centers = bins[:-1] + (bins[1:] - bins[:-1]) / 2
    ax.bar(bin_centers, counts, width=(bins[1] - bins[0]), alpha=0.6, label='Observed')

    # Set labels and limits
    plt.xlabel('A2 Values')
    plt.ylabel('Probability Density')
    ax.legend()
    #ax.set_yscale('log')
    ax.set_xlim([min_val, max_val])
    #ax.set_ylim([1e-5, 10**(0.5)])

    # Calculate skew and kurtosis
    data_skew = skew(A2)
    data_kurtosis = kurtosis(A2)
    print("Skew:", data_skew)
    print("Kurtosis:", data_kurtosis)
    
    # Check if the p-value is less than significance level (e.g., 0.05)
    significance_level = 0.05

    if type == 'norm':
        if p_value < significance_level:
            print("The generated PDF is significantly different from the Gaussian distribution.")
        else:
            print("The generated PDF is not significantly different from the Gaussian distribution.")
    else:
        if p_value_exp < significance_level:
            print("The generated PDF is significantly different from the Exponential distribution.")
        else:
            print("The generated PDF is not significantly different from the Exponential distribution.")
        
        if p_value_power < significance_level:
            print("The generated PDF is significantly different from the Power Law distribution.")
        else:
            print("The generated PDF is not significantly different from the Power Law distribution.")

    plt.show()
    return fig


def min_max_align(arr1, arr2, use_min: bool = False) -> np.array:
    ''' Function to align two arrays based on their shared values using
        either the location of the min or max to trim. Note that the var
        name 'time' is used, but values don't have to be in time format
        
    Args:
        use_min: turn on to use min to control triming instead of max'''

    if use_min:
        min_val_arr1 = min(arr1)
        min_val_arr2 = min(arr2)

        if min_val_arr2 <= min_val_arr1:
            time_values_t = np.array(arr1)
            indices2 = np.searchsorted(arr2, time_values_t, side='right') - 1
            indices2 = np.clip(indices2, 0, len(arr2)-1)
            time2 = arr2[indices2]
            time1 = time_values_t
            indices1 = np.searchsorted(arr1, time2)

        elif min_val_arr1 < min_val_arr2:
            time_values_t = np.array(arr2)
            indices1 = np.searchsorted(arr1, time_values_t, side='right') - 1
            indices1 = np.clip(indices1, 0, len(arr1)-1)
            time1 = arr1[indices1]
            time2 = time_values_t
            indices2 = np.searchsorted(arr2, time1)
        else:
            indices1 = np.searchsorted(arr1, arr2, side='right') - 1
            indices1 = np.clip(indices1, 0, len(arr1)-1)
            indices2 = np.searchsorted(arr2, arr1, side='right') - 1
            indices2 = np.clip(indices2, 0, len(arr2)-1)
            time1 = arr1[indices1]
            time2 = arr2[indices2]

    else:
        max_val_arr1 = max(arr1)
        max_val_arr2 = max(arr2)

        if max_val_arr2 >= max_val_arr1:
            time_values_t = np.array(arr1)
            indices2 = np.searchsorted(arr2, time_values_t, side='left') - 1
            indices2 = np.clip(indices2, 0, len(arr2)-1)
            time2 = arr2[indices2]
            time1 = time_values_t
            indices1 = np.searchsorted(arr1, time2, side='left')

        elif max_val_arr1 > max_val_arr2:
            time_values_t = np.array(arr2)
            indices1 = np.searchsorted(arr1, time_values_t, side='left') - 1
            indices1 = np.clip(indices1, 0, len(arr1)-1)
            time1 = arr1[indices1]
            time2 = time_values_t
            indices2 = np.searchsorted(arr2, time1, side='left')

        else:
            indices1 = np.searchsorted(arr1, arr2, side='left') - 1
            indices1 = np.clip(indices1, 0, len(arr1)-1)
            indices2 = np.searchsorted(arr2, arr1, side='left') - 1
            indices2 = np.clip(indices2, 0, len(arr2)-1)
            time1 = arr1[indices1]
            time2 = arr2[indices2]
    
    return time1, time2, indices1, indices2


def retrieve_region(ds, lon: any = None, lat: any = None, min_time: any = None, max_time: any = None,
                    month_range: any = None, year_range: any = None,
                    ids: any = None, deplat: any = None, deplon: any = None, deptime: any = None,
                    wmos: any = None, full: bool = True, debug = False, not_ids: any = None,
                    dt_range: any = None, vertices: any = None, clip_vertices: bool = False) -> xr.Dataset:
    '''Subset a ragged array drifter dataset for a region in space and time using coords
    to subset by obs and variables to subset by trajs. Modified from the Philippe et al.
    NOAA GDP code found at: https://github.com/Cloud-Drift/earthcube-meeting-2022/tree/main 
    
    Args:
        ds: xarray Dataset
        lon: longitude slice of the subregion (min,max)
        lat: latitude slice of the subregion (min,max)
        max_time: max time of the subregion
        min_time: min time of the subregion
        dt_range: controls length of trajectories (i.e., how many seconds to take from each traj)
        vertices: A list containing lon/lat points to subset based on a custom shape
        ids: ID of drifters
        not_ids: ids to be excluded from the dataset
        deplat: deployment latitude slice
        deplon: deployment longitude slice
        deptime: deployment time slice
        wmos: WMO IDs of drifters NOTE: it is best to use IDs instead of WMOs as some WMO's repeat in the ds
    
    Returns: 
        ds_subset: Dataset of the subregion
    '''

    # Initialize the masks for each dimension
    mask = np.ones(ds.dims['obs'], dtype=bool)
    mask_id = np.ones(ds.dims['traj'], dtype=bool)

    if vertices is not None: # Mask using a custom shape

        # Create a dataframe of lat/lon points from the ds
        latitude_array = ds.latitude.values
        longitude_array = ds.longitude.values
        geometry = gpd.points_from_xy(longitude_array, latitude_array)
        gdf = gpd.GeoDataFrame(geometry=geometry, crs=pyproj.CRS("epsg:4326"))

        polygon = gpd.GeoDataFrame({'geometry': [Polygon(vertices)]})
        polygon = polygon.unary_union
        #polygon = polygon.scale(1.1, 1.1) # Scale if you want the shape to be bigger or smaller
        intersects = gdf.intersects(polygon, align=False)

        if clip_vertices:
            mask &= ~intersects.values                                                                                                                                                                                                                
        else:
            mask &= intersects.values

    if lon: # Mask longitude points
        mask &= (ds.longitude >= lon[0]) & (ds.longitude <= lon[1])

    if lat: # Mask latitude points
        mask &= (ds.latitude >= lat[0]) & (ds.latitude <= lat[1])

    if min_time is not None: # Mask min_time points
        mask &= (ds.time >= np.datetime64(min_time))

    if max_time is not None: # Mask max_time points
        mask &= (ds.time <= np.datetime64(max_time))

    if month_range is not None: # Mask months
        mask &= (ds.time.dt.month.isin(month_range))

    if year_range is not None: # Mask years
        mask &= (ds.time.dt.year.isin(year_range))

    if ids is not None: # Mask IDs
        mask &= np.isin(ds.ids, ids)

    if not_ids is not None:  # remove unwanted ids
        mask &= ~np.isin(ds.ids, not_ids)

    if dt_range is not None: # Mask out observations based on a set length (e.g., 100 days)
        ds = get_dt(ds) # Sets the dt parameter for each trajectory
        mask &= (ds.dt >= dt_range[0]) & (ds.dt <= dt_range[1])

    if debug: # Random debug info because I keep breaking stuff
        print('Debug info if using ids: this is a vector of the matches found ' + str(np.isin(ds.coords['ids'][ds.traj.isin(ids)], ids)))
        print('Debug info: this is a vector of matches found considering all criteria ' + str(mask))
        print('Debug info: total number of matches: ' + str(sum(mask > 0)))
        print('Debug info: total number removed: ' + str(sum(mask <= 0)))

    ## MASKING TRAJ DIMENSION ##
    # update traj mask using the ID numbers from the obs mask
    mask_id &= ds.ID.isin(np.unique(ds.ids[mask])) # I moved this to the start of the traj masking because I think it would return all 1s
                                                   # if I had it at the end and had no conditions in the obs dim
    if deplat: # Mask DEPLOY latitude points
        mask_id &= (ds.deploy_lat >= deplat[0]) & (ds.deploy_lat <= deplat[1])

    if deplon: # Mask DEPLOY longitude points
        mask_id &= (ds.deploy_lon >= deplon[0]) & (ds.deploy_lon <= deplon[1])

    if deptime: # Mask DEPLOY time points
        mask_id &= (ds.deploy_date >= np.datetime64(deptime[0])) & (ds.deploy_date <= np.datetime64(deptime[1]))

    if wmos is not None: # Mask WMOs
        mask_id &= np.isin(ds.WMO, wmos)

    mask_id &= ds.rowsize > 1 # Removes trajectories with less than two obs

    # update obs mask using the traj mask with consideration of the 'full' option
    full_mask = ds.ids.isin(ds.ID[mask_id]) # Determines if fulll trajectories should be used for drifters
                                            # that come into the domain - full trajs will align w/ traj_in
    domain_mask = mask
    domain_mask &= ds.ids.isin(ds.ID[mask_id])

    if full: # cut trajectories where they leave the domain (if you don't want full trajs)
        ds_subset = ds.isel(obs=np.where(full_mask)[0], traj=np.where(mask_id)[0])

    else:
        ds_subset = ds.isel(obs=np.where(domain_mask)[0], traj=np.where(mask_id)[0])

    if (len(ds_subset.time.values) <= 0):
        print(ds_subset.time.values)
        print('No valid drifters based on supplied arguments.')
        #print(ds_subset)
    else:
        ## Store ds wmos in a temp array and use it to find where WMOs don't exist
        #wmo_blank = ds_subset.WMO.values
        #zerowmos = np.where(ds_subset.WMO.values == 0) # Another method may be to check for uniqueness of WMOs
        #                                               # This way repeated WMOs can be replaced too
        ## Set missing WMOs to drifter IDs
        #if any(n != 0 for n in zerowmos):  
        #    generated_wmos = ds_subset.ID.values[zerowmos[0]]
        #    print('Generated wmos',generated_wmos,len(generated_wmos))
        #    wmo_blank[zerowmos[0]] = generated_wmos

        #    #Replace WMOs
        #    ds_subset.WMO.values = wmo_blank

        # Reset rowsizes to match new ds
        ds_subset.rowsize.values = get_rowsize(ds=ds_subset)

        # Calculate dt to match new ds <- for when masking only part of a trajectory
        ds_subset = get_dt(ds=ds_subset)

    return ds_subset


def drogue_subset(ds) -> xr.Dataset:
    ''' Takes a drifter dataset and returns two new datasets which contain 
        1: Observations where the drogue was present
        2: Observations where the drogue was not present'''

    traj_idx, trajsum = get_traj_index(ds)

    # Initialize the masks for each dimension
    drogue_mask = []
    undrogue_mask = []
    drogue_mask_traj = np.ones(ds.dims['traj'], dtype=bool)
    undrogue_mask_traj = np.ones(ds.dims['traj'], dtype=bool)

    # Mask values based on the date of when the drogue is lost
    for i in range(len(ds.traj)):
        undrogued_i = ds.time[slice(traj_idx[i], trajsum[i])].values >= ds.drogue_lost_date[i].values
        undrogue_mask.append(undrogued_i)
        drogue_mask.append(~undrogued_i)

    drogue_mask = np.concatenate(drogue_mask)
    undrogue_mask = np.concatenate(undrogue_mask)
    drogue_mask_traj &= ds.ID.isin(np.unique(ds.ids[drogue_mask].values))
    undrogue_mask_traj &= ds.ID.isin(np.unique(ds.ids[undrogue_mask].values))

    # Apply masks to create new datasets
    undrogued_ds = ds.isel(obs=np.where(undrogue_mask)[0], traj=np.where(undrogue_mask_traj)[0])
    drogued_ds = ds.isel(obs=np.where(drogue_mask)[0], traj=np.where(drogue_mask_traj)[0])

    # Reset rowsizes and get dts
    undrogued_ds.rowsize.values = get_rowsize(ds=undrogued_ds)
    drogued_ds.rowsize.values = get_rowsize(ds=drogued_ds)
    undrogued_ds = get_dt(undrogued_ds)
    drogued_ds = get_dt(drogued_ds)

    return drogued_ds, undrogued_ds


def segments(ds, Tl = 10):
    ''' Generates a new dataset (with a number of vars dropped) which contains
        segmented drifter trajectories every Tl days'''

    Tl_sec = Tl * 24 * 3600 # Convert days to seconds

    traj_idx, trajsum = get_traj_index(ds)

    def find_and_duplicate_arrays(original_array, divisor, error = 24*3600):
        indices = np.where(original_array % divisor == 0)[0]
        duplicated_arrays = [original_array[i:] for i in indices]

        return indices, duplicated_arrays
    
    # Create a list to store records
    records = []

    # Iterate over trajectories
    for i, traj in enumerate(ds.ID.values.astype(str)):
        # Select traj data
        ids = (ds.ids[slice(traj_idx[i], trajsum[i])].values).astype(str)
        lats = ds.latitude[slice(traj_idx[i], trajsum[i])].values
        lons = ds.longitude[slice(traj_idx[i], trajsum[i])].values
        times = ds.time[slice(traj_idx[i], trajsum[i])].values
        ssts = ds.sst[slice(traj_idx[i], trajsum[i])].values
        ve = ds.ve[slice(traj_idx[i], trajsum[i])].values
        vn = ds.vn[slice(traj_idx[i], trajsum[i])].values
        dts = ds.dt[slice(traj_idx[i], trajsum[i])].values
        #rowsize = ds.rowsize[i].values
        #deploy_lat = ds.deploy_lat[i].values
        #deploy_lon = ds.deploy_lon[i].values
        
        # Append the original record
        #original_record = { 
        #    "ID": xr.DataArray([traj], dims={"traj": [1]}),
        #    "deploy_lon": xr.DataArray([deploy_lon], dims={"traj": [1]}),
        #    "deploy_lat": xr.DataArray([deploy_lat], dims={"traj": [1]}),
        #    "latitude": xr.DataArray([lats], dims={"traj": [1], "obs": np.arange(rowsize)}),
        #    "longitude": xr.DataArray([lons], dims={"traj": [1], "obs": np.arange(rowsize)}),
        #    "time": xr.DataArray([times], dims={"traj": [1], "obs": np.arange(rowsize)}),
        #    "sst": xr.DataArray([ssts], dims={"traj": [1], "obs": np.arange(rowsize)}),
        #    "ve": xr.DataArray([ve], dims={"traj": [1], "obs": np.arange(rowsize)}),
        #    "vn": xr.DataArray([vn], dims={"traj": [1], "obs": np.arange(rowsize)})   
        #}
        #records.append(original_record)

        # Create new trajs
        segment_indices, _ = find_and_duplicate_arrays(dts, divisor=Tl_sec)

        # Iterate over segment_indices
        for j, seg_idx in enumerate(segment_indices):

            len_obs = len(ids[seg_idx:])

            # Append each segment record
            segment_record = { 
                "ID": xr.DataArray([traj + str((j + 1) / 1000)], dims={"traj": [1]}),
                "deploy_lon": xr.DataArray([lons[seg_idx:][0]], dims={"traj": [1]}),
                "deploy_lat": xr.DataArray([lats[seg_idx:][0]], dims={"traj": [1]}),
                "latitude": xr.DataArray([lats[seg_idx:]], dims={"traj": [1], "obs": np.arange(len_obs)}),
                "longitude": xr.DataArray([lons[seg_idx:]], dims={"traj": [1], "obs": np.arange(len_obs)}),
                "time": xr.DataArray([times[seg_idx:]], dims={"traj": [1], "obs": np.arange(len_obs)}),
                "sst": xr.DataArray([ssts[seg_idx:]], dims={"traj": [1], "obs": np.arange(len_obs)}),
                "ve": xr.DataArray([ve[seg_idx:]], dims={"traj": [1], "obs": np.arange(len_obs)}),
                "vn": xr.DataArray([vn[seg_idx:]], dims={"traj": [1], "obs": np.arange(len_obs)})   
            }
            records.append(segment_record)

    # Convert the list of records to an Awkward Array
    #print(records)
    #print(segment_record)
    #test_ds = xr.Dataset(records[0])
    #print(test_ds)
    new_ds = create_ragged_arr(records).to_xarray()
    
    #print(new_ds)
    print('---------------- SEGMENTING DONE -----------------------')

    # Get dts and traj_cols
    #new_ds.rowsize.values = get_rowsize(ds=new_ds) # Shouldn't need this but adding in for safety
    new_ds = get_dt(ds=new_ds)
    #new_ds = get_traj_cols(ds=new_ds)
    return new_ds


def get_enter_index(ds, ds_tmp: any = None,
                    lat: any = None, lon: any = None, vertices: any = None): # NOTE: DEPRECATED
    ''' Find the time where a trajectory first enters a specified region
        within a larger region. Function returns a mask-like array that
        can be passed to the i_region function NOTE: i_region functions not currently working'''
    
    # Create a subset if it doesn't exist already
    if (lat is not None) or (lon is not None) or (vertices is not None):
        ds_tmp = retrieve_region(ds=ds, lat=lat, lon=lon, vertices=vertices, full=False)

    traj_idx, trajsum = get_traj_index(ds_tmp) # Trajectories within small region
    traj_idx2, trajsum2 = get_traj_index(ds) # Trajectories within original region

    # Determine difference in rowsizes
    match_index = np.isin(ds.ID.values, ds_tmp.ID.values)
    #print(len(match_index), match_index)
    #print(ds.ID[match_index].values)
    start_dif = traj_idx2[match_index] - traj_idx
    tail_dif = trajsum2[match_index] - trajsum
    print(len(tail_dif), tail_dif)
    print(len(start_dif), start_dif)
    rs_dif = ds.rowsize[match_index].values - ds_tmp.rowsize.values
    rs_dif = rs_dif - tail_dif

    trajsum_test = np.cumsum(ds.rowsize[match_index].values, dtype=np.int64)
    # Adds a zero at the start (repressents first drifter index)
    traj_index_full = np.insert(trajsum_test,0,0)
    # Deletes last index to re-align size
    traj_idx_test = np.delete(traj_index_full, len(traj_index_full)-1)

    # Create the mask
    mask = np.ones(ds.dims['obs'], dtype=bool)
    
    adjusted_idx = traj_idx + rs_dif
    print(len(rs_dif), rs_dif)
    print(len(traj_idx), traj_idx)
    print(len(adjusted_idx), adjusted_idx)
    adjustedsum = trajsum + rs_dif
    slc = slice(adjusted_idx, trajsum2[match_index])
    print(slc)
    mask[slc] = 1
    print(mask)

    ## Extract time valuues to align arrays
    #t1 = ds_tmp.time[traj_idx].values
    #t2 = ds.time[traj_idx2].values
    #
    ## Align arrays using min_max_align
    #time1, time2, indices1, indices2 = min_max_align(t1, t2, use_min=True)

    return mask


def i_region(ds, indices): # NOTE: DEPRECATED
    ''' Subsets a drifter dataset using a supplied array of indices.
        This is intended to function faster than the regular "retrieve_region"
        function and also allow for subsetting based on individual trajectories.'''
    
    # Initialize the masks for each dimension
    # NOTE: Supplied indices are only in the obs dimension
    mask = indices
    mask_id = np.ones(ds.dims['traj'], dtype=bool)
    mask_id &= ds.ID.isin(np.unique(ds.ids[mask]))

    # Update masks for each dimension
    domain_mask = mask
    domain_mask &= ds.ids.isin(ds.ID[mask_id])

    # Apply mask
    ds_subset = ds.isel(obs=np.where(domain_mask)[0], traj=np.where(mask_id)[0])

    # Reset rowsizes to match new ds
    ds_subset.rowsize.values = get_rowsize(ds=ds_subset)

    # Calculate dt to match new ds <- for when masking only part of a trajectory
    ds_subset = get_dt(ds=ds_subset)

    return ds_subset


def driftplot(ds, ids: bool = True, sst: bool = False, velocity: bool = False,
              time: bool = False, labs: bool = False, startpt: bool = True,
              borders: bool = True, domain: list = None, pinfo: bool = False,
              traj_cmap: any = None, traj_norm: any = None, set_dpi: int = 150,
              veast: bool = False, vnorth: bool = False, A2: bool = False,
              x2: bool = False, y2: bool = False) -> plt.figure:
    '''plots SVP drifter trajectories and particular attributes
    
    Args:
        ds: xarray Dataset
        sst: switch for sst plot
        velocity: switch for velocity plot
        labs: switch for labels at points
        startpt: switch for starting location
        borders: switch for coastline
        domain: extent to be plotted in form (xmin, xmax,ymin,ymax)
        A2, x2, y2: absolute dispersion in the combined, zonal, and meridional directions
    
    Returns: 
        fig: figure of drifter trajectories
    '''
    # Return drifter info 
    if pinfo:
        print('Drifter IDs ' +str(ds.ID.values))
        print('Number of observations ' +str(ds.sizes['obs']))
        print('Number of drifters ' +str(len(ds.ID.values)))
        print(ds.deploy_date.values)

    # Create figure 
    fig = plt.figure(figsize=(6,6), dpi=set_dpi)
    ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.02, axes_class=plt.Axes)
    
    # Plot either sst, time since deployment or velocity   
    # Plot time since deployment
    if time:
        pcm = ax.scatter(ds.longitude, ds.latitude,
                         s=0.5, c=(ds.dt / 3600 / 24), #Change to datetime?
                         transform=ccrs.PlateCarree(), cmap=mpl.colormaps['Blues'])
        cb = fig.colorbar(pcm, cax=cax)
        cb.set_label('Time (days)')

    # Plot velocity at each point
    elif velocity:
        pcm = ax.scatter(ds.longitude, ds.latitude,
                         s=0.5, c=np.sqrt(ds.ve**2+ds.vn**2),
                         transform=ccrs.PlateCarree(), cmap=cmocean.cm.speed,
                         vmin=0)
        cb = fig.colorbar(pcm, cax=cax)
        cb.set_label('Velocity Magnitude [m/s]')
    
    elif vnorth: # Meridional
        pcm = ax.scatter(ds.longitude, ds.latitude,
                s=0.5, c=(ds.vn.values),
                transform=ccrs.PlateCarree(), cmap=cmocean.cm.speed)
        cb = fig.colorbar(pcm, cax=cax)
        cb.set_label('Velocity Magnitude [m/s]')
    
    elif veast: # Latitudinal
        pcm = ax.scatter(ds.longitude, ds.latitude,
                s=0.5, c=(ds.ve.values),
                transform=ccrs.PlateCarree(), cmap=cmocean.cm.speed)
        cb = fig.colorbar(pcm, cax=cax)
        cb.set_label('Velocity Magnitude [m/s]')
    
    # Plot absolute dispersion at each point
    elif A2:
        pcm = ax.scatter(ds.longitude, ds.latitude,
                         s=0.5, c=np.log10(ds.A2),
                         transform=ccrs.PlateCarree(), cmap=cmocean.cm.speed,
                         vmin=0, vmax=7)
        cb = fig.colorbar(pcm, cax=cax)
        cb.set_label('Log10 Total Dispersion (m^2/s^2)')

    elif x2: # zonal
        pcm = ax.scatter(ds.longitude, ds.latitude,
                s=0.5, c=np.log10(ds.disp_x2),
                transform=ccrs.PlateCarree(), cmap=cmocean.cm.speed,
                vmin=0, vmax=7)
        cb = fig.colorbar(pcm, cax=cax)
        cb.set_label('Log10 Zonal Dispersion (m^2/s^2)')

    elif y2: # meridional
        pcm = ax.scatter(ds.longitude, ds.latitude,
                s=0.5, c=np.log10(ds.disp_y2),
                transform=ccrs.PlateCarree(), cmap=cmocean.cm.speed,
                vmin=0, vmax=7)
        cb = fig.colorbar(pcm, cax=cax)
        cb.set_label('Log10 Meridional Dispersion (m^2/s^2)')

    # Plot sst at each point
    elif sst:
        pcm = ax.scatter(ds.longitude, ds.latitude,
                         marker='.', s=0.5, c=ds.sst-273.15,
                         transform=ccrs.PlateCarree(), cmap=cmocean.cm.thermal,
                         vmin=-2, vmax=35)
        cb = fig.colorbar(pcm, cax=cax)
        cb.set_label('Temperature [C]')

    # Otherwise plot by id (default)
    else:
        # Create a color map using the generated colors
        #random_seed = 69  # Change seed to change colours
        #np.random.seed(random_seed)
        #random_colors = np.random.rand(len(ds.traj), 3)

        # Create a colormap using the consistent colors
        #cmap = colors.ListedColormap(random_colors)
        
        pcm = ax.scatter(ds.longitude, ds.latitude,
                         s=0.5, marker='o', color=ds.traj_cols.values,
                         )
        cbar = plt.colorbar(mappable=plt.cm.ScalarMappable(cmap=traj_cmap, norm=traj_norm),
                            cax=cax, orientation="vertical",
                            ticks=np.arange(len(np.unique(ds.ID.values))) + 0.5)
        cbar.set_label("ID")
            
        cbar.ax.set_yticklabels(np.unique(ds.ID.values)) # Using unique here and above to remove repeated labels
                                                         # in the dataframe if it is supplied
        cbar.remove()

    # Add ID labels to scatter if given the option
    if labs:
        for i, txt in enumerate(ds.ID.values):
            ax.annotate(txt, (ds.deploy_lon[i],ds.deploy_lat[i]))

    # Add starting points to drifter trajs
    if startpt:
        ax.scatter(ds.deploy_lon, ds.deploy_lat,
                   s=0.5, marker="$\u25EF$", c='black'
                   )

   # Add land and coastlines to plot 
    if borders:
        ax.add_feature(cfeature.LAND, facecolor='bisque', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.25, zorder=1)

    # Set the displayed region
    if domain:
        ax.set_extent(domain, crs=ccrs.PlateCarree())
        ax.set_xticks(np.arange(domain[0],domain[1],40), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(domain[2],domain[3],15), crs=ccrs.PlateCarree())
    else:
        ax.set_extent((145,165,-45,-20), crs=ccrs.PlateCarree())
        ax.set_xticks(np.arange(145,165,5), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-45,-20,5), crs=ccrs.PlateCarree())
    
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())


def binplot(ds, A2: bool = False, sst: bool = False, sigma = 0.25,
            time: bool = False, x2: bool = False, y2: bool = False,
            borders: bool = True, domain = [145,165,-45,-20], res = 0.5,
            set_dpi: int = 150, velocities: bool = False, c_levels = 5) -> plt.figure:

    # Create a gridded space to bin values
    #lon = np.linspace(domain[0], domain[1], np.abs(domain[0] - domain[1])**2)
    #lat = np.linspace(domain[2], domain[3], np.abs(domain[2] - domain[3])**2)
    lat = np.arange(domain[2], domain[3] + res, res)
    lon = np.arange(domain[0], domain[1] + res, res)
    x_c = np.convolve(lon, [0.1, 0.1], mode='valid')
    y_c = np.convolve(lat, [0.1, 0.1], mode='valid')

    # Total dispersion
    if A2:
        # Interpolate to fill gaps
        points = np.column_stack((ds.longitude.values.flatten(), ds.latitude.values.flatten()))
        values = (abs(ds.d_A2.values.flatten())) # Convert from m^2 / s to km^2 / day

        lon_interp = lon
        lat_interp = lat
        xi, yi = np.meshgrid(lon_interp, lat_interp)

        A2_interp = griddata(points, values, (xi, yi), method='nearest')

        # Apply Gaussian smoothing
        A2_interp = gaussian_filter(A2_interp, sigma=sigma, mode='reflect', truncate=0.25)
        #A2_smooth = A2_interp
        #A2_smooth = (xi, yi)

        # Plotting

        # get 1st and 99th percentiles of values to plot to get a useful range for the colorscale
        #v1,v2 = np.nanpercentile(A2_interp,[1,90])

        fig = plt.figure(dpi=set_dpi)
        ax = fig.add_subplot(projection=ccrs.PlateCarree())

        # Contour lines
        contour_levels = c_levels
        contour_plot = ax.contour(xi, yi, A2_interp, levels=contour_levels, colors='black',
                                  linewidths=np.linspace(0.01, 1.25, contour_levels))

        # Add contour labels
        ax.clabel(contour_plot, inline=True, fontsize=5, zorder=9)

        # Filled contour plot
        pcm = ax.imshow(A2_interp, extent=[domain[0], domain[1], domain[2], domain[3]], cmap=cmaps.NCV_jet,
                        norm=colors.SymLogNorm(linthresh=0.1), origin='lower', interpolation='None')

        # gridlines and labels
        #gl = ax.gridlines(color='k', linewidth=0.1, linestyle='-',
        #                  xlocs=np.arange(domain[0], domain[1], 1),
        #                  ylocs=np.arange(domain[2], domain[3], 1),
        #                  draw_labels=True)
        #gl.top_labels = False
        #gl.right_labels = False
        ## add land and coastline
        ax.add_feature(cfeature.LAND, facecolor='grey', zorder=10)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.25, zorder=10)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05, axes_class=plt.Axes)
        cb = fig.colorbar(pcm, cax=cax)
        cb.ax.set_ylabel('Absolute Diffusivity (km^2/day)')
        # Set the displayed region
        ax.set_extent(domain, crs=ccrs.PlateCarree())
    
    # Zonal dispersion
    elif x2:
        # Interpolate to fill gaps
        points = np.column_stack((ds.longitude.values.flatten(), ds.latitude.values.flatten()))
        values = np.log10(ds.d_x2.values.flatten())
        lon_interp = np.linspace(domain[0], domain[1], np.abs(domain[0] - domain[1])**2)
        lat_interp = np.linspace(domain[2], domain[3], np.abs(domain[2] - domain[3])**2)
        xi, yi = np.meshgrid(lon_interp, lat_interp)
        x2_interp = griddata(points, values, (xi, yi), method='linear')

        # Apply Gaussian smoothing
        x2_smooth = gaussian_filter(x2_interp, sigma=sigma)

        # Plotting
        fig = plt.figure(dpi=set_dpi)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        pcm = ax.imshow(x2_smooth, extent=[domain[0], domain[1], domain[2], domain[3]], cmap=cmocean.cm.speed,
                        origin='lower')

        # gridlines and labels
        gl = ax.gridlines(color='k', linewidth=0.1, linestyle='-',
                          xlocs=np.arange(domain[0], domain[1], 1),
                          ylocs=np.arange(domain[2], domain[3], 1),
                          draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        # add land and coastline
        ax.add_feature(cfeature.LAND, facecolor='grey', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.25, zorder=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05, axes_class=plt.Axes)
        cb = fig.colorbar(pcm, cax=cax)
        cb.ax.set_ylabel('Log10 Zonal Dispersion (m^2/s^2)')
        # Set the displayed region
        ax.set_extent(domain, crs=ccrs.PlateCarree())

    # Meridional dispersion
    elif y2:
        # Interpolate to fill gaps
        points = np.column_stack((ds.longitude.values.flatten(), ds.latitude.values.flatten()))
        values = np.log10(ds.d_y2.values.flatten())
        lon_interp = np.linspace(domain[0], domain[1], np.abs(domain[0] - domain[1])**2)
        lat_interp = np.linspace(domain[2], domain[3], np.abs(domain[2] - domain[3])**2)
        xi, yi = np.meshgrid(lon_interp, lat_interp)
        y2_interp = griddata(points, values, (xi, yi), method='linear')

        # Apply Gaussian smoothing
        y2_smooth = gaussian_filter(y2_interp, sigma=sigma)

        # Plotting
        fig = plt.figure(dpi=set_dpi)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        pcm = ax.imshow(y2_smooth, extent=[domain[0], domain[1], domain[2], domain[3]], cmap=cmocean.cm.speed,
                        origin='lower')

        # gridlines and labels
        gl = ax.gridlines(color='k', linewidth=0.1, linestyle='-',
                        xlocs=np.arange(domain[0], domain[1], 1),
                        ylocs=np.arange(domain[2], domain[3], 1),
                        draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        # add land and coastline
        ax.add_feature(cfeature.LAND, facecolor='grey', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.25, zorder=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05, axes_class=plt.Axes)
        cb = fig.colorbar(pcm, cax=cax)
        cb.ax.set_ylabel('Log10 Meridional Dispersion (m^2/s^2)')
        # Set the displayed region
        ax.set_extent(domain, crs=ccrs.PlateCarree())

    # Temperature
    elif sst:
        sst_bin = stats.binned_statistic_2d(ds.longitude, 
                        ds.latitude,
                        ds.sst,
                        statistic=np.nanmean,
                        bins=[lon, lat])

        # get 1st and 99th percentiles of values to plot to get a useful range for the colorscale
        v1,v2 = np.nanpercentile(sst_bin.statistic.T,[1,99])

        fig = plt.figure(dpi=set_dpi)
        ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
        pcm = ax.pcolormesh(x_c, y_c, 
                            sst_bin.statistic.T-273.15, 
                            cmap=cmocean.cm.thermal,
                            vmin=-2, vmax=32)

        # gridlines and labels
        gl = ax.gridlines(color='k', linewidth=0.1, linestyle='-',
                          xlocs=np.arange(domain[0], domain[1], 1),
                          ylocs=np.arange(domain[2], domain[3], 1),
                          draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False

        # add land and coastline
        ax.add_feature(cfeature.LAND, facecolor='grey', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.25, zorder=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05, axes_class=plt.Axes)
        cb = fig.colorbar(pcm, cax=cax)
        cb.ax.set_ylabel('Temperature (Degrees C)')

        # Set the displayed region
        ax.set_extent(domain, crs=ccrs.PlateCarree())

    else:
        # Eastward velocity binned
        veast = stats.binned_statistic_2d(ds.longitude, 
                                        ds.latitude,
                                        ds.ve,
                                        statistic=np.nanmean, 
                                        bins=[lon, lat])

        # Northward velocity binned
        vnorth = stats.binned_statistic_2d(ds.longitude, 
                                    ds.latitude,
                                    ds.vn,
                                    statistic=np.nanmean,
                                    bins=[lon, lat])
        
        velocities = {0: veast, 1: vnorth}

        # Create a figure and individual subplots
        fig, ax1 = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        ax2 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())

        # Plot eastward velocity
        pcm1 = ax1.pcolormesh(x_c, y_c,
                            velocities[0].statistic.T,
                            cmap=cmocean.cm.speed,
                            )
        ax1.gridlines(color='k', linewidth=0.1, linestyle='-',
                    xlocs=np.arange(domain[0], domain[1], 1),
                    ylocs=np.arange(domain[2], domain[3], 1),
                    draw_labels=True)
        ax1.gridlines.top_labels = False
        ax1.gridlines.right_labels = False
        ax1.add_feature(cfeature.LAND, facecolor='grey', zorder=1)
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.25, zorder=1)
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="3%", pad=0.05, axes_class=plt.Axes)
        cb1 = fig.colorbar(pcm1, cax=cax1)
        cb1.set_label('Eastward Velocity (m/s)')
        ax1.set_extent(domain, crs=ccrs.PlateCarree())

        # Plot northward velocity
        pcm2 = ax2.pcolormesh(x_c, y_c,
                            velocities[1].statistic.T,
                            cmap=cmocean.cm.speed,
                            )
        ax2.gridlines(color='k', linewidth=0.1, linestyle='-',
                    xlocs=np.arange(domain[0], domain[1], 1),
                    ylocs=np.arange(domain[2], domain[3], 1),
                    draw_labels=True)
        ax2.gridlines.top_labels = False
        ax2.gridlines.right_labels = False
        ax2.add_feature(cfeature.LAND, facecolor='grey', zorder=1)
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.25, zorder=1)
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="3%", pad=0.05, axes_class=plt.Axes)
        cb2 = fig.colorbar(pcm2, cax=cax2)
        cb2.set_label('Northward Velocity (m/s)')
        ax2.set_extent(domain, crs=ccrs.PlateCarree())

    return fig


def ibm_plot(ds, borders: bool = True, labs: bool = False, startpt: bool = True,
            number: bool = True, domain: any = [140,170,-15,-50]) -> plt.figure:
    '''Plots trajectories from an IBM simulation file
    
    Args:
        ds: xarray Dataset
        velocity: switch for velocity plot (not implemented yet)
        labs: switch for labels at points
        startpt: switch for starting location
        borders: switch for coastline
        number: switch to set colours of trajectories to their corresponding ID 
        domain: extent to be plotted in form (xmin, xmax,ymin,ymax)
    
    Returns: 
        fig: figure of drifter trajectories'''

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.02, axes_class=plt.Axes)

    if number:
        pcm = ax.scatter(ds['lon'], ds['lat'],
                         s=0.5, c=ds.origin_marker.values,
                         transform=ccrs.PlateCarree(),
                         vmin=0, vmax=9)
        cb = fig.colorbar(pcm, cax=cax)
        cb.set_label('Drifter number')

    # Add ID labels to scatter if given the option
    if labs:
        print(ds.trajectory.values)
        print(ds.dropna(dim='time').lon[0:,0].values)
        for i, txt in enumerate(ds.trajectory.values):
            ax.annotate(txt,(ds.dropna(dim='time').lon[0:,0][i].values,
                        ds.dropna(dim='time').lat[0:,0][i].values))

    # Add starting points to drifter trajs
    if startpt:
        ax.scatter(ds.lon[0:,0],
                    ds.lat[0:,0],
                    s=2, c='black', transform=ccrs.PlateCarree())

   # Add land and coastlines to plot 
    if borders:
        ax.add_feature(cfeature.LAND, facecolor='grey', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.25, zorder=1)

    # Set the displayed region
    if domain:
        ax.set_extent(domain, crs=ccrs.PlateCarree())
        ax.set_xticks(np.arange(domain[0],domain[1]), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(domain[2],domain[3]), crs=ccrs.PlateCarree())
    
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    return fig


def drift_frames(ds: any = None, IBM: any = None, IBM_2: any = None, domain: any = (145,165,-45,-20),
                 plot_int: int = 1, ts: int = 1, labs: any = ['SVP drifters', 'Simulated drifters'],
                 framedir: str = 'C:/Users/z5493451/OneDrive - UNSW/Documents/Data/Drifters/animations',
                 duration: int = None, traj_cmap: any = None, traj_norm: any = None):
    '''Function to create frames for animations of drifter trajectories. Works for both GDP and IBM datasets.
        If both a real (i.e., GDP) and IBM dataset are provided, both will be plotted. If multiple IBM datasets
        are provided, they will both be plotted. 
        
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
        time_delta = datetime.timedelta(seconds=seconds)
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
            traj_cmap, traj_norm = traj_cmap(traj=ds.traj)
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
    
    # MAKE THE ANIMATION #
    frame_files = sorted(glob.glob(f"{framedir}/frame_*.png"))
    images = [Image.open(filename) for filename in frame_files]

    images[0].save(outputdir, save_all=True, append_images=images[1:], duration=100, loop=0)
    print("Animation created from frames.")


def start_sep(ds, startpts: dict = None, op_type: str = 'square') -> pd.DataFrame:
    ''' Uses drifter metadata to calculate the starting distance between all possible drifter pairs'''

    # Create dataframe of drifter start locations and times
    if startpts is None:
        startpts = drift_meta(ds)
    
    # Calculate the number of points to compare
    #num_pts = len(startpts['start_lat'])

    # Initialize the distance output array
    startdists = {}

    # Calculate the distance and time between each possible pair
    for i, j in combinations(startpts['ID'], 2):
        # Calculate the start separation of all possible pairs
        dist = distance.distance((startpts['start_lat'][i], startpts['start_lon'][i]),
                                (startpts['start_lat'][j], startpts['start_lon'][j])).km
        startdists[(i, j)] = {'start_sep': dist, 'start_dt': 0}
        startdists[(j, i)] = {'start_sep': dist, 'start_dt': 0} if op_type == 'square' else {'start_sep': np.nan, 'start_dt': np.nan}

        # Calculate the start time differences of all possible pairs
        timediff = np.timedelta64(startpts['start_time'][i] - startpts['start_time'][j])
        startdists[(i, j)]['start_dt'] = timediff
        startdists[(j, i)]['start_dt'] = timediff if op_type == 'square' else np.nan
    
    # Convert to dataframe
    df = pd.DataFrame.from_dict(startdists, orient='index').dropna(axis=0)

    # Reset the index and rename the pair column
    df = df.reset_index().rename(columns={'index': 'pair'})
    df.index.name = 'row_index'

    # Find and add drifter IDs to df
    ID0 = ds.ID[np.int32(df['level_0'])].values
    ID1 = ds.ID[np.int32(df['level_1'])].values
    df['ID0'] = np.full_like(df['level_0'], fill_value=ID0, dtype=np.int64())
    df['ID1'] = np.full_like(df['level_1'], fill_value=ID1, dtype=np.int64())  

    return df


def pair_sep(ds, ibm: any = None, IDs : list = None,
            min_time: any = None, debug : bool = False) -> pd.DataFrame:
    ''' Module to calculate the distance between drifters at each time recorded
        required input is a list of IDs (currently only 2 useable for comparison)
        and a combined xarray dataset of their positions etc. - note that the module
        will also filters out the drifters from a large ds like the GDP.v2.00.ncf file.

        Also works for comparison of real vs. virtual (i.e., IBM) drifters by supplying
        the IBM argument which is an array in the format generated by an IBM such as
        Opendrift

        Args:
            min_time: is used when calculating chance pair separations to set start time
            ds: drifter dataset. This should contain full trajectories to avoid breaks in dt 
    '''

    # Find corresponding index for IDs
    j = np.int32(np.where(ds.ID == IDs[0])[0])

    #Create subset containing info for first drifter
    ds_temp1 = retrieve_region(ds, ids = IDs[0], min_time=min_time[0], full=False)

    # Comparrison of subsets based on IBM or not
    if not ibm: # Use a second drifter for comparisson if not comparing between virtual drifters

        # Create subset for second drifter
        ds_temp2 = retrieve_region(ds, ids = IDs[1], min_time=min_time[1], full=False)

        #print('temp2 length',len(ds_temp2.obs.values))
        #print('temp1 length',len(ds_temp1.obs.values))
        #print('temp1 rowsizes',ds_temp1.rowsize.values)

        # Stop if there are eroneous trajs
        if len(ds_temp2.traj.values) == 0:
            print('Not a valid pair - check the time ranges of drifters perhaps?')
            return None
        else:
            pass

        # Using min time as a condition to check if we should calc for chance pairs as it should
        # only be supplied when chance pairs are supplied
        if min_time is not None:
            time_values, time2, indices_temp1, indices_temp2 = min_max_align(ds_temp1.dt.values, ds_temp2.dt.values, use_min=False)

    elif ibm: #subset if comparing between virtual and real drifters
       
        # subset the ibm dataset based on an origin marker
        ds_temp2 = ibm.where(ibm.trajectory == j+1, drop=True).dropna(dim='time')
        
        #Initialize distances and times for OP dataframe
        time_values = np.array(ds_temp2.time.values, dtype='datetime64[s]')
        #distances = np.zeros(len(time_values), dtype=float)

        # Find indices of time_values in ds_temp1
        indices_temp1 = np.unique(np.searchsorted(ds_temp1.time.values.astype('datetime64[s]'), time_values))
        #print(indices_temp1, len(indices_temp1))

        if debug:
            print('Original IBM obs length: ',len(ibm.time.values))
            print('Physical drifter start lat: ',ds_temp1.latitude[0].values,'virtual drifter start lat: ',ds_temp2.lat[0,0].values)
            print('Physical drifter start lon: ',ds_temp1.longitude[0].values,'virtual drifter start lat: ',ds_temp2.lon[0,0].values)
            print('Physical drifter start time: ',ds_temp1.time[0].values,'virtual drifter start time: ',ds_temp2.time[0].values)
    
    # Initialize distance array
    distances = np.full(len(time_values), fill_value=np.nan, dtype=float)

    # Initialize zonal array
    dist_zonal = np.full(len(time_values), fill_value=np.nan, dtype=float)

    # Initialize meridional array
    dist_meridional = np.full(len(time_values), fill_value=np.nan, dtype=float)

    # Extract latitude and longitude arrays from ds_temp1
    lat_temp1 = ds_temp1.latitude.values
    lon_temp1 = ds_temp1.longitude.values

    # Get latitude and longitude values corresponding to time_values from ds_temp1
    lat_values_temp1 = lat_temp1[indices_temp1]
    lon_values_temp1 = lon_temp1[indices_temp1]

    # Create array of ids for drifter 1
    ids_1 = np.full_like(distances, fill_value=ds_temp1.ID[0].values, dtype=np.int64)

    if not ibm: # Need conditions since output ds for drifters and ibm have different lon/lat variable names
        lat_temp2 = ds_temp2.latitude.values
        lon_temp2 = ds_temp2.longitude.values

        # Get latitude and longitude values corresponding to time_values from ds_temp2
        lat_values_temp2 = lat_temp2[indices_temp2]
        lon_values_temp2 = lon_temp2[indices_temp2]

        # Create array of ids for drifter 2
        ids_2 = np.full_like(distances, fill_value=ds_temp2.ID[0].values, dtype=np.int64)

    elif ibm is not None:
        lat_values_temp2 = ds_temp2.lat.values[0]
        lon_values_temp2 = ds_temp2.lon.values[0]

        # Create array of ids for drifter 2
        ids_2 = np.full_like(distances, fill_value=ds_temp2.trajectory.values, dtype=np.int64)

    # Create a matrix of latitude and longitude pairs
    d1 = np.column_stack((lat_values_temp1, lon_values_temp1))
    d2 = np.column_stack((lat_values_temp2, lon_values_temp2))
    
    # Create list of dts based on 'time_values'
    dt = (time_values - time_values[0]).astype('timedelta64[s]').view('int64')

    if debug:
        if ibm:
            print('Drifter number',int(np.where(ds.ID == IDs[0])[0]),
                   'and virtual drifter: ', ds_temp2.trajectory[0].values-1)
        else:
            print('1st Drifter number:',int(np.where(ds.ID == IDs[0])[0]),
                   'and 2nd drifter: ', int(np.where(ds.ID == IDs[1])[0]))
        print('Time info:',time_values, len(time_values))
        print('Distance info:',distances, len(distances))
        print('Deltatimes: ',dt)
        print(d1, len(d1))
        print(d2, len(d2))

    # Calculate distances using matrix operations
    for i in range(len(time_values)):
        dist = distance.distance(d1[i], d2[i]).km
        zonal_i = distance.distance(d1[i],(d2[i][0],d1[i][1])).km
        meridional_i = distance.distance(d1[i],(d1[i][0],d2[i][1])).km
        distances[i] = dist if dist <= 60000 else np.nan # If statement removes eronious values (i.e., dist shouldn't exceed Earth's circ)
        dist_zonal[i] = zonal_i
        dist_meridional[i] = meridional_i

    # Calculate D^2 for relative dispersion
    D2 = distances**2
    Z2 = dist_zonal**2
    M2 = dist_meridional**2

    # Create new unique ID for each pair
    ids_1 = ids_1.astype(str)
    ids_2 = ids_2.astype(str)
    ID_marker = [str1 + str2 for str1, str2 in zip(ids_1, ids_2)]
    traj_cols_first = np.full_like(distances, fill_value=ds_temp1.traj_cols[0].values, dtype=str)
    traj_cols_second = np.full_like(distances, fill_value=ds_temp2.traj_cols[0].values, dtype=str)

    # Create a DataFrame with distances and time values
    df_dist = pd.DataFrame(data={'ID': ids_1, 'comp_ID': ids_2, 'ID_marker': ID_marker, 'time_first': time_values,
                                 'time_second': time2, 'latitude_first': lat_values_temp1, 'longitude_first': lon_values_temp1,
                                 'latitude_second': lat_values_temp2, 'longitude_second': lon_values_temp2,
                                 'dt': dt, 'sep_distance': distances, 'traj_cols_first': traj_cols_first,
                                 'traj_cols_second': traj_cols_second, 'D2': D2, 'dist_zonal': dist_zonal,
                                 'dist_meridional': dist_meridional, 'Z2': Z2, 'M2': M2},
                        columns=['ID', 'comp_ID', 'ID_marker', 'time_first', 'time_second', 'latitude_first', 'longitude_first',
                                 'latitude_second', 'longitude_second', 'dt', 'sep_distance', 'traj_cols_first','traj_cols_second',
                                 'D2', 'dist_zonal', 'dist_meridional', 'Z2', 'M2'])
    return df_dist


def find_pairs(ds, ddist0: float = 5, dt0: float = 6, start_dists: any = None) -> pd.DataFrame:
    ''' Short module to find drifter pairs based on their initial separation and start times.
        in the future I will try implement the capability of finding pairs at any time'''
    
    # Find drifter pairs based on start posn. and times
    if start_dists is None:
        start_dists = start_sep(ds, op_type='triangular') #First create df of start locs and times
    else:
        pass

    pairs = start_dists.where((start_dists['start_sep'] <= ddist0) &
                              (np.abs(start_dists['start_dt']) <= np.timedelta64(dt0,'h'))).dropna()
    pairs[['ID0','ID1']] = pairs[['ID0','ID1']].astype(np.int64())
    return pairs


def cohort_pair_sep(ds, ddist0: float = 5, dt0: float = 6, start_dists: any = None,
                    pairs: any = None, cp_df: any=None) -> pd.DataFrame:
    ''' Module to calculate the pair separations (D) of a cohort of drifter pairs
        for use in calculating relative dispersal and FSLE. 

        Args:
        ds: xarray dataset in a ragged array format like that of the GDP2.00 file
        ddist0: the starting seperation distance used as a cut-off to identify a drifter pair
        dt0: the starting time difference used as a cut-off to identify a drifter pair
        ts: timestep (in seconds) between observations - currently not used
        debug: switch for debug info  
        
        Hopefully returns: D and D2 of each trajectory at each dt so that these can be
                            averaged over all pairs to calc relative dispersal'''
    
    if cp_df is not None:
        # Set the pairs dataframe to the chance pairs dataframe if it is provided
        pairs = cp_df
        ID1 = pairs.ID_first.values
        ID2 = pairs.ID_second.values
        min_time = True # Switches min time on so that it is used in the following 'pair_sep' function 
    else:
        min_time = False # Min time is not needed for initial pairs since both trajs start at the same time
        # Run the initial pair calculation if chance_pairs aren't given
        print('No chance pair dataframe (cp_df) provided: running for initial pairs only.')
        if (start_dists is None) and (pairs is None):
            pairs = find_pairs(ds, ddist0, dt0)
        elif (pairs is None) and (start_dists is not None):
            pairs = find_pairs(ds, ddist0, dt0, start_dists=start_dists)   
        else:
            pass
        # Find the corresponding identifiers of the drifters
        ID1 = pairs.ID0.values
        ID2 = pairs.ID1.values

    # Calculate pair separations
    dfs = []
    for i in tqdm(range(len(pairs.index.values))):
        #print('Run:',i,'comp between drifters:',list([ID1[i],ID2[i]]))
        start_times = [pairs['time_first'].values[i], pairs['time_second'].values[i]]
        pairsep = pair_sep(ds, IDs=list([ID1[i],ID2[i]]), min_time=start_times if min_time else None)
        dfs.append(pairsep)
    cps = pd.concat(dfs)
    
    # Filter out pairs that are too far apart in time
    def delta_filter(df, max_delta=7200, max_dt=100 * 24 * 3600):
        # Calculate the time difference
        df['time_diff'] = (df['time_second'] - df['time_first']).abs()
        # Filter rows where the time difference is greater than 1 hour
        filtered_df = df[df['time_diff'] <= max_delta]
        filtered_df = filtered_df[filtered_df['dt'] <= max_dt]
        # Drop the 'time_diff' column if it's no longer needed
        filtered_df = filtered_df.drop(columns=['time_diff'])
        return filtered_df[filtered_df['D2'] > 0]

    return delta_filter(cps)


def displacement_unused(ds, ID, debug : bool = False) -> dict:
    ''' Module to calculate the displacement distance (i.e., absolute dispersal)
        of a drifter from its starting location at each time recorded.
        Required input is a ID (or some identifier linked to the array of posn.)
        and a combined xarray dataset of its positions etc. - note that the module
        will also filters out the drifter from a large ds like the GDP.v2.00.ncf file

        returns a dictionary that can be converted to a pd.dataframe - I use a dict 
        so that I can iterate over a large number of drifters and so that the OP is
        useable with my other funcs 
    '''
 
    #Create subset containing info of the drifter
    ds_temp1 = retrieve_region(ds, ids = ID)

    #Initialize distances and times for OP dataframe
    distances = np.zeros_like(ds_temp1.time, dtype=float)
    times = np.zeros_like(ds_temp1.time, dtype='datetime64[s]')

    if debug: #Debug stuff because I always break things
        print('Drifter number',int(np.where(ds.ID == ID)[0]))
        print('Time info:',times, len(times))
        print('Distance info:',distances, len(distances))

    # Extract latitude and longitude arrays from ds_temp1
    lat_temp1 = ds_temp1.latitude.values
    lon_temp1 = ds_temp1.longitude.values

    # Extract time values
    time_values = np.array(ds_temp1.time.values, dtype='datetime64[s]')

    # Create a matrix of latitude and longitude pairs
    d1 = np.column_stack((lat_temp1, lon_temp1))

    # Calculate distances using matrix operations
    for i in range(len(time_values)):
        dist = distance.distance(d1[0],d1[i]).km #Distance between the start point and the ith point
        distances[i] = dist if dist <= 30000 else np.nan #Error checking - distance shouldn't exceed Earth's circumference

    # Create a DataFrame with distances and time values
    ids = np.full_like(distances, fill_value=ID)
    df_dist = {'ID':ids,
        'lat':lat_temp1,
        'lon':lon_temp1,
        'time':time_values,
        'sep_distance':distances}
    
    return df_dist


def cohort_displace(ds, tau = 20, debug : bool = False, sub_mean: bool = True) -> xr.Dataset:
    ''' Calculates cohort displacements (i.e., absolute dispersal)
        
        args:
        ds: xarray dataset containing traj and obs dim
        tau: max time lag (in days) to integrate to - this will calculate the
            displacement for a point in tau many days.'''
    
    # Define function to calculate geodesic distance between consecutive points
    def per_chunk_displace(latitude, longitude, dt, threshold: float = 20, pt_disp: bool = False):
        initial_point = (latitude[0], longitude[0]) # Should really change order to match other funcs (lon, lat)
        points = list(zip(longitude, latitude))
        displacements = []
        zonal_components = []
        meridional_components = []
        pt_displacements = []
        pt_zonals = []
        pt_meridionals = []
        tau_idx = []

        for i in range(len(longitude)): # Calculate dispersion and its components setting lat or lon constant or changing
            lon, lat = points[i]            
            point = (lat, lon)

            displacement = distance.geodesic(initial_point, point).km # Total dispersion
            zonal_component = distance.geodesic(initial_point, (lat, initial_point[1])).km # Zonal dispersion
            meridional_component = distance.geodesic(initial_point, (initial_point[0], lon)).km # Merid dispersion
            
            # Calculate single point displacements (for mapping)
            if pt_disp:
                threshold_sec = threshold * 24 * 3600
                points_inv = points[::-1]
                #print('points')
                #print(points)
                #print('points_inv')
                #print(points_inv)
                dt_inv = dt[::-1]
                #print('dt')
                #print(dt)
                #print('dt_inv')
                #print(dt_inv)
                dt_i = dt_inv[i]
                tau_idx = find_index(dt_inv, threshold_sec, dt_i)
                #print('tau_idx')
                #print(tau_idx)

                if (np.isnan(tau_idx)):  # Set pt displacement to nan where tau exceeds length of time series
                    pt_displacement = np.nan # (i.e., first tau days are cut from the time series)
                    pt_zonal = np.nan
                    pt_meridional = np.nan
                else:
                    point_inf_lon, point_inf_lat = points_inv[i] # lat(-t|t_0, x), lon(-t|t_0, x)
                    pt2 = (point_inf_lat, point_inf_lon)
                    inv_lon, inv_lat = points_inv[tau_idx]
                    pt1 = (inv_lat, inv_lon)  #lat(t_0), lon(t_0)
                    pt_displacement = distance.geodesic(pt1, pt2).km
                    pt_zonal = distance.geodesic(pt1, (pt2[0], pt1[1])).km
                    pt_meridional = distance.geodesic(pt1, (pt1[0], pt2[1])).km
            else:
                pt_displacement = np.nan
                pt_zonal = np.nan
                pt_meridional = np.nan

            displacements.append(displacement)
            zonal_components.append(zonal_component)
            meridional_components.append(meridional_component)
            pt_displacements.append(pt_displacement)
            pt_zonals.append(pt_zonal)
            pt_meridionals.append(pt_meridional)
           
        return displacements, zonal_components, meridional_components, pt_displacements[::-1], pt_zonals[::-1], pt_meridionals[::-1]
        
    # Create a dataframe containing grided velocity averages for entire ds
    # (this is used to remove the background flow when calculating dispersion)
    if sub_mean:
        points = np.column_stack((ds.longitude.values.flatten(), ds.latitude.values.flatten()))
        total_velocity = np.sqrt(ds.ve.values ** 2 + ds.vn.values ** 2)
        grid_df = grid_data(points=points, u=ds.ve.values.flatten(),
                            v=ds.vn.values.flatten(), total=total_velocity)
    
    # Calculate displacements and then dispersion
    traj_idx, trajsum = get_traj_index(ds)
    disp = []
    zonal = []
    meridional = []
    v_prime = []
    u_prime = []
    total_prime = []
    pt_d = []
    pt_x = []
    pt_y = []

    for i in range(len(ds.traj)):  #Calc displacement for traj i
        lat = ds.latitude[slice(traj_idx[i], trajsum[i])].values
        lon = ds.longitude[slice(traj_idx[i], trajsum[i])].values
        u = ds.ve[slice(traj_idx[i], trajsum[i])].values
        v = ds.vn[slice(traj_idx[i], trajsum[i])].values
        dt = ds.dt[slice(traj_idx[i], trajsum[i])].values
        dt_diff_i = np.diff(dt)
        dt_diff_i = np.insert(dt_diff_i, dt[0], 0)

        if tau is not None:
            displacement_i, zonal_i, meridional_i, pt_displacement_i, pt_zonal_i, pt_meridional_i = per_chunk_displace(lat,lon,dt,tau,pt_disp=True)
        else:
            displacement_i, zonal_i, meridional_i, pt_displacement_i, pt_zonal_i, pt_meridional_i = per_chunk_displace(lat,lon,dt,pt_disp=False)
        
        if sub_mean:
            # Subtract 'background' flow by finding where each lat/lon falls in the gridded data
            tot = total_velocity[slice(traj_idx[i], trajsum[i])]
            gridded_tree = cKDTree(np.column_stack((grid_df['lon'], grid_df['lat']))) #KDTree for the gridded data
            _, indices = gridded_tree.query(np.column_stack((lon, lat)), k=1) #Find closest pts
            closest_totals = grid_df['total'].iloc[indices].values #Corresponding values for closest pts
            closest_u = grid_df['u'].iloc[indices].values
            closest_v = grid_df['v'].iloc[indices].values
    
            # Apply mean correction
            # NOTE on converting m / s to km / h: since we calculate displacement in km per n (usually 3600 - an hour) sec
            # initially we need to either convert it to m / s or convert the velocities to km / n seconds. The choice of 
            # which conversion we do is relatively arbitrary, however we want to keep disspersion values in km for ploting
            # time series (dispersal_plot fnc) and diffusivity in m / s for mapping. As such, we convert velocities, first
            # to kilometers a second (e.g., closest totals * 1000) and then to km per n second by multiplying by the time
            # difference array
            displacement_i = displacement_i - ((closest_totals) * 1000 / dt_diff_i)
            zonal_i = zonal_i - ((closest_u) * 1000 / dt_diff_i)
            meridional_i = meridional_i - ((closest_v) * 1000 / dt_diff_i)
            v_prime_i = v - closest_v
            u_prime_i = u - closest_u
            total_prime_i = tot - closest_totals

            # Return NaN if the mean does not exist (i.e., if they have been filtered where N < threshold)
            nan_indices = np.isnan(closest_totals)
            displacement_i[nan_indices] = np.nan
            zonal_i[nan_indices] = np.nan
            meridional_i[nan_indices] = np.nan
            u_prime_i[nan_indices] = np.nan
            v_prime_i[nan_indices] = np.nan
            total_prime_i[nan_indices] = np.nan

            u_prime.append(u_prime_i)
            v_prime.append(v_prime_i)
            total_prime.append(total_prime_i)

        disp.append(displacement_i)
        zonal.append(zonal_i)
        meridional.append(meridional_i)
        pt_d.append(pt_displacement_i)
        pt_y.append(pt_zonal_i)
        pt_x.append(pt_meridional_i)

    pt_d = np.concatenate(pt_d)
    pt_y = np.concatenate(pt_y)
    pt_x = np.concatenate(pt_x)
    pt_d = xr.DataArray(pt_d, coords={'ids': ds.ids}, dims=['obs'])
    pt_x = xr.DataArray(pt_x, coords={'ids': ds.ids}, dims=['obs'])
    pt_y = xr.DataArray(pt_y, coords={'ids': ds.ids}, dims=['obs'])

    ret = np.concatenate(disp)
    ret2 = np.concatenate(zonal)
    ret3 = np.concatenate(meridional)

    # Create a new DataArray for total dispersion
    displacement = xr.DataArray(ret, coords={'ids': ds.ids}, dims=['obs']) #Used to have obs coord too idk?
    A2 = displacement ** 2
    time_diff = np.diff(ds.dt.values)
    A2_diff = np.diff(A2) # Multiply by 1000 to convert to m / s
    d_A2 = 0.5 * (A2_diff / time_diff) # Diffusivity
    # Prepend 0 to the derivative array to account for the initial condition at t0
    d_A2 = np.insert(d_A2, 0, 0)
    d_A2 = xr.DataArray(d_A2, coords={'ids': ds.ids}, dims=['obs'])
    
    # Create a new DataArray for meridional dispersion
    #disp_x = xr.DataArray(ret3, coords={'ids': ds.ids, 'obs': ds.obs}, dims=['obs'])
    disp_x = xr.DataArray(ret3, coords={'ids': ds.ids}, dims=['obs'])
    disp_x2 = disp_x**2
    x2_diff = np.diff(disp_x2) # Multiply by 1000 to convert to m / s
    d_x2 = 0.5 * (x2_diff / time_diff) # Zonal Diffusivity
    d_x2 = np.insert(d_x2, 0, 0)
    d_x2 = xr.DataArray(d_x2, coords={'ids': ds.ids}, dims=['obs']) 
    
    # Create a new DataArray for zonal dispersion
    #disp_y = xr.DataArray(ret2, coords={'ids': ds.ids, 'obs': ds.obs}, dims=['obs'])
    disp_y = xr.DataArray(ret2, coords={'ids': ds.ids}, dims=['obs'])
    disp_y2 = disp_y**2
    y2_diff = np.diff(disp_y2) # Multiply by 1000 to convert to m / s
    d_y2 = 0.5 * (y2_diff / time_diff) # Meridional Diffusivity
    d_y2 = np.insert(d_y2, 0, 0)
    d_y2 = xr.DataArray(d_y2, coords={'ids': ds.ids}, dims=['obs'])

    if sub_mean:
        ret4 = np.concatenate(u_prime)
        ret5 = np.concatenate(v_prime)
        ret6 = np.concatenate(total_prime)
        u_prime = xr.DataArray(ret4, coords={'ids': ds.ids}, dims=['obs'])
        v_prime = xr.DataArray(ret5, coords={'ids': ds.ids}, dims=['obs'])
        total_prime = xr.DataArray(ret6, coords={'ids': ds.ids}, dims=['obs'])
        #d_prime = cKDTree(np.column_stack((grid_df['lon'], grid_df['lat']))) 

        # Add new data to original dataset
        new_ds = ds.assign(displacement=displacement, A2=A2, disp_x=disp_x,
                           disp_y=disp_y, disp_x2=disp_x2, disp_y2=disp_y2,
                           d_A2=d_A2, d_x2=d_x2, d_y2=d_y2,
                           total_prime=total_prime, v_prime=v_prime,
                           u_prime=u_prime, pt_x=pt_x, pt_y=pt_y, pt_d=pt_d
                           )
    else:
        # Add new data to original dataset
        new_ds = ds.assign(displacement=displacement, A2=A2, disp_x=disp_x,
                           disp_y=disp_y, disp_x2=disp_x2, disp_y2=disp_y2,
                           d_A2=d_A2, d_x2=d_x2, d_y2=d_y2, pt_x=pt_x, pt_y=pt_y, pt_d=pt_d
                           )

    return new_ds  # Return the new dataset


def lagrangian_autocorr(velocities_list, u, v):
    """
    Calculate the Lagrangian autocorrelation for a list of velocity arrays.
    
    Args:
        velocities_list (list): A list of numpy arrays, where each array represents velocities recorded for a single trajectory.
    
    Returns:
        list of numpy.ndarray: List of Lagrangian autocorrelation values over all trajectories for different time lags.
    """
    autocorrs = []
    zonals = []
    merids = []
    
    for i, velocities in tqdm(enumerate(velocities_list)):
        N = len(velocities)
        mean_velocity = np.nanmean(velocities)
        mean_zonal = np.nanmean(u[i])
        mean_merid = np.nanmean(v[i])

        centered_velocities = velocities - mean_velocity
        centered_zonal = u[i] - mean_zonal
        centered_merids = v[i] - mean_merid
        
        autocorr = np.correlate(centered_velocities, centered_velocities, mode='full') / (N * np.var(velocities))
        autocorr = autocorr[N - 1:]
        autocorrs.append(autocorr)

        zonal = np.correlate(centered_zonal, centered_zonal, mode='full') / (N * np.var(u[i]))
        zonal = zonal[N - 1:]
        zonals.append(zonal)

        merid = np.correlate(centered_merids, centered_merids, mode='full') / (N * np.var(v[i]))
        merid = merid[N - 1:]
        merids.append(merid)
    
    return autocorrs, zonals, merids


def get_velocities(ds, tinf = 200.0, primes: bool = True):
    traj_idx, trajsum = get_traj_index(ds=ds)

    velocities = []
    velocity_devs = []
    u_devs = []
    v_devs = []
    traj_dts = []

    for i in range(len(ds.traj.values)): # Seperate trajectories into individual arrays
        if primes:
            devs_tot_i = ds.total_prime[slice(traj_idx[i], trajsum[i])].values
            devs_u = ds.u_prime[slice(traj_idx[i], trajsum[i])].values
            devs_v = ds.v_prime[slice(traj_idx[i], trajsum[i])].values
        velocities_tot_i = ds.v_total[slice(traj_idx[i], trajsum[i])].values
        traj_dt_i = ds.dt[slice(traj_idx[i], trajsum[i])].values / 3600 / 24

        # Cut each traj at tinf or set to na if not found
        tinf_idx = find_index(arr=traj_dt_i, x=tinf, threshold=0)
        if np.isnan(tinf_idx):
            continue
        else:
            velocities_tot_i = velocities_tot_i[:tinf_idx]
            traj_dt_i = traj_dt_i[:tinf_idx]
            velocities.append(velocities_tot_i)
            traj_dts.append(traj_dt_i)
            if primes:
                devs_tot_i = devs_tot_i[:tinf_idx]
                devs_u = devs_u[:tinf_idx]
                devs_v = devs_v[:tinf_idx]
                velocity_devs.append(devs_tot_i)
                u_devs.append(devs_u)
                v_devs.append(devs_v)

    return velocities, velocity_devs, u_devs, v_devs, traj_dts


def plot_autocorrelations(velocities_list, u_list, v_list, times_list, max_x=20):
    autocorrs, zonal, merid = lagrangian_autocorr(velocities_list, u_list, v_list)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(4,7), dpi=125)

    # Plot individual autocorrelations
    for autocorr, times in zip(autocorrs, times_list):
        if max_x is not None:
            mask = times <= max_x
            autocorr = autocorr[:np.sum(mask)]
            times = times[mask]
        ax.plot(times, autocorr, alpha=0.5, lw=0.5, color='gainsboro')

    # Flatten times and autocorrelations
    all_times = np.concatenate(times_list)
    all_autocorrs = np.concatenate(autocorrs)
    all_zonal = np.concatenate(zonal)
    all_merid = np.concatenate(merid)

    # Calculate mean autocorrelation at each unique time point
    unique_x_values = np.unique(all_times)
    mean_cor_values = []
    mean_u_cor_values = []
    mean_v_cor_values = []
    for x_value in unique_x_values:
        mean_cor_value = np.nanmean(all_autocorrs[all_times == x_value])
        mean_cor_values.append(mean_cor_value)

        mean_u_cor_value = np.nanmean(all_zonal[all_times == x_value])
        mean_u_cor_values.append(mean_u_cor_value)

        mean_v_cor_value = np.nanmean(all_merid[all_times == x_value])
        mean_v_cor_values.append(mean_v_cor_value)

    mean_cor_values = np.array(mean_cor_values)
    mean_u_cor_values = np.array(mean_u_cor_values)
    mean_v_cor_values = np.array(mean_v_cor_values)

    # Apply max_x limit to the mean values if specified
    if max_x is not None:
        mask = unique_x_values <= max_x
        unique_x_values = unique_x_values[mask]
        mean_cor_values = mean_cor_values[mask]
        mean_u_cor_values = mean_u_cor_values[mask]
        mean_v_cor_values = mean_v_cor_values[mask]

    # Plot mean autocorrelation
    ax.plot(unique_x_values, mean_cor_values, color='black', linestyle='solid', label='Mean', linewidth=1)
    ax.plot(unique_x_values, mean_u_cor_values, color='black', linestyle='dashed', label='Mean U', linewidth=1)
    ax.plot(unique_x_values, mean_v_cor_values, color='black', linestyle='dotted', label='Mean V', linewidth=1)

    # Find the first crossing point of the mean autocorrelation with zero
    zero_crossing_index = np.where(np.diff(np.sign(mean_cor_values)))
    print(zero_crossing_index)
    zero_crossing_time = min(unique_x_values[zero_crossing_index])
    print(zero_crossing_time)

    # Add a vertical line at the zero-crossing time
    ax.axvline(x=zero_crossing_time, color='steelblue', linestyle='--', label='Zero Crossing', zorder=11)

    # Add a horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle=':', zorder=10)

    # Show y ticks every two ticks
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Plot decorations
    ax.set_xlabel('Time lag (days)', fontsize=10)
    ax.set_ylabel('Autocorrelation', fontsize=10)
    ax.tick_params(labelsize=10)
    ax.grid(linestyle='dotted')
    #plt.show()

    # Calculate the integral up to the first zero crossing
    int_times = []
    zonal_ints = []
    merid_ints = []
    max_idx = np.where(unique_x_values == max_x)[0][0]
    for i in range(max_idx):
        int_time = np.trapz(mean_cor_values[:i], unique_x_values[:i])
        zonal_int = np.trapz(mean_u_cor_values[:i], unique_x_values[:i])
        merid_int = np.trapz(mean_v_cor_values[:i], unique_x_values[:i])
        zonal_ints.append(zonal_int)
        merid_ints.append(merid_int)
        int_times.append(int_time)

    int_times = np.array(int_times)
    zonal_ints = np.array(zonal_ints)
    merid_ints = np.array(merid_ints)

    # Plot the integral times
    #fig2, ax2 = plt.subplots(figsize=(4,3.5), dpi=125)
    ax2.plot(unique_x_values[:max_idx], int_times, lw=1.5, color='black')
    ax2.plot(unique_x_values[:max_idx], zonal_ints, lw=1.5, color='black', linestyle='dashed')
    ax2.plot(unique_x_values[:max_idx], merid_ints, lw=1.5, color='black', linestyle='dotted')
    ax2.axhline(y=max(int_times), color='steelblue', linestyle='--', zorder=10)

    # Show y ticks every two ticks
    ax2.yaxis.set_major_locator(MultipleLocator(0.25))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.set_ylim(0, 1.5)

    ax2.set_xlabel('Time lag (days)', fontsize=10)
    ax2.set_ylabel('Timescale (days)', fontsize=10)
    ax2.tick_params(labelsize=10)
    ax2.grid(linestyle='dotted')
    plt.show()


    #return fig, fig2
    return fig


def find_breakpoints(ds, relative: bool = False, guess: any = None, num_breakpoints: int = 2):
    ''' Function to fit a piecewise model to a set of x and y data.
        Note, that if relative is specified it will fit an exponential
        as the first line segment followed by a number of power laws.'''
    
    # Initialize OP
    d_funcs = {}

    # Transform input arrays
    if relative: # Takes natural log of y values to fit exponential
        y = ds.D2.values
        x = ds.dt.values / 3600 / 24

        valid_indices = np.logical_and(np.logical_and(x > 0, x < 200), y > 0)
        x = x[valid_indices]
        y = y[valid_indices]
        x1 = np.log10(x)
        y1 = np.log10(y)

        # Perform broken-stick regression
        model = PiecewiseLinFit(x1, y1)
        if guess is not None:
            breakpoints = model.fit_guess(guess)
        else:
            breakpoints = model.fit(num_breakpoints)

        # Get the coefficient estimates
        coefficients = model.slopes
        intercepts = model.intercepts
        #print(coefficients)
        #print(breakpoints)
        #print(coefficients2)
        #print(breakpoints2)

        # Fit exponential using natural log of y
        y2 = np.log(y)
        
        # Perform broken-stick regression
        model2 = PiecewiseLinFit(x, y2)
        if guess is not None:
            breakpoints2 = model2.fit_guess(guess)
        else:
            breakpoints2 = model2.fit(num_breakpoints)
        #breakpoints2 = np.insert(breakpoints2, 0, 0) # Add zero for start point
        # Get the coefficient estimates
        coefficients2 = model2.slopes
        intercepts2 = model2.intercepts

        for i in range(len(breakpoints)-1):
            if (i == 0): 
                d_funcs[f'exp:f{i}'] = {'range': (min(x), breakpoints2[1]),
                                    'power': coefficients2[0],
                                    'coef': intercepts2[0]}
            else:
                d_funcs[f'f{i}'] = {'range': (10 ** breakpoints[i], 10 ** breakpoints[i+1]),
                                    'power': coefficients[i],
                                    'coef': 10 ** intercepts[i]}

    else:
        # Takes log10 to fit power laws
        x = ds.dt.values / 3600 / 24
        y = ds.A2.values
        valid_indices = np.logical_and(np.logical_and(x > 0, x < 200), y > 0)
        valid_x = x[valid_indices]
        valid_y = y[valid_indices]
        x = np.log10(valid_x)
        y = np.log10(valid_y)

        # Perform broken-stick regression
        model = PiecewiseLinFit(x, y)
        if guess is not None:
            breakpoints = model.fit_guess(guess)
        else:
            breakpoints = model.fit(num_breakpoints)
        #breakpoints = np.insert(breakpoints, 0, 0) # Add zero for start point

        # Get the coefficient estimates
        coefficients = model.slopes
        intercepts = model.intercepts
        #print(coefficients)
        #print(breakpoints)

        for i in range(len(breakpoints)-1):
            d_funcs[f'f{i}'] = {'range': (10 ** breakpoints[i], 10 ** breakpoints[i+1]),
                        'power': coefficients[i],
                        'coef': 10 ** intercepts[i]}
            
        print(d_funcs)
    return d_funcs


def fit_trends(x: any = None, y: any = None, plot: bool = True,
               exp: bool = False, is_powerlaw: bool = True):
    ''' Perform regression to find the best fitting power-law (or other provided func)
        for dispersal data.
        
        Args:
            plot: if true, will plot the observed values (x and y) and the best fit line
            '''

    # Test to see if the data are distributed in a power-law distribution
    #fit = Fit(y, xmin=min(y))
    #is_powerlaw = fit.distribution_compare('power_law', 'exponential',
    #                                       normalized_ratio=True)[0] > 0
    #print('Powerlaw test is: ',is_powerlaw)
    #is_powerlaw = True

    # Initializing because valid_x seems to vanish for some reason
    if exp:
        is_powerlaw = False
        # Take log of y for regression
        y_log = np.log(y)
        valid_x = x

        # Perform linear regression on the log-transformed data
        m, c, r_value, p_value, std_err = linregress(x, y_log)
        y_fit_log = m * x + (c - 1.5) # +1 is used to adjust the line so that it doesn't overrlap the data
        y_fit = np.exp(y_fit_log)

    if is_powerlaw:
        # Filter out zero, negative, and invalid values
        valid_indices = np.logical_and(x > 0, y > 0)
        valid_x = x[valid_indices]
        valid_y = y[valid_indices]
        log_x_vals = np.log10(valid_x)
        log_y_vals = np.log10(valid_y)

        # Fit regression line on loglog data 
        m, c, r_value, p_value, std_err = linregress(log_x_vals, log_y_vals)
        log_y_fit = m*log_x_vals + (c + 1.5) # +1 is used to adjust the line so that it doesn't overrlap the data
        y_fit = 10 ** log_y_fit

    # Store fitted values in a dataframe
    fit_frame = pd.DataFrame(np.column_stack([valid_x, y_fit]))
    results = {'slope': [m],
               'intercept': [c],
               'r_value': [r_value],
               'p_value': [p_value],
               'std_err': [std_err]} 
    results_df = pd.DataFrame(results)
    print(results_df)

    if plot and is_powerlaw:
        # Plot observed and fitted values
        plt.scatter(log_x_vals, log_y_vals, label='Observed', s=0.5)
        plt.plot(log_x_vals, log_y_fit, color='red', label='Fitted')
        # Add labels and legend
        plt.xlabel('X Values')
        plt.ylabel('Y Values')
        plt.legend()
        # Annotate the slope on the plot
        slope_annotation = f'Power: {m:.2f}'
        plt.annotate(slope_annotation, xy=(0.1, 0.8), xycoords='axes fraction', fontsize=12)
    return results_df, fit_frame
    #else:
    #    print('Data does not fit a power-law distribution')
    #    return None, None


def dispersal_plot(ds: any = None, df: any = None, scatter: bool = True,
                   components: bool = True, total: bool = True,
                   title = '', log: bool = True, max_dt = 100,
                   disp_funcs: dict = {}, best_fit: bool = False,
                   cmap: any = None, norm: any = None,
                   ci: bool = True, theoretical: bool = False) -> plt.figure:
    ''' Creates plots of absolute (A2) and relative (D2) dispersal.
        Also has the option to plot mean trendlines or to not plot
        scatter points (i.e., only plot trendlines). The function is
        also used to conduct a Kolmogorov-Smirnov test on the observed
        data to compare it with a number of theoretical regimes which
        can be supplied by the user using the disp_funcs parameter
        NOTE: This function is very messy and needs better implementation for
            dataframe and dataset separation.
        
        Args:
            ds: xarray dataset of drifter trajectories with absolute dispersal values,
                if absolute dispersal has not been calc'd yet, run the ds through the
                cohort_displace function first
            df: a dataframe containing drifter pair separations (relative dispersal),
                if this AND a ds is supplied the function will plot relative dispersal,
                otherwise it will plot absolute dispersal
            scatter: set this to false to remove observation scaatter points from the plot
            components: set this to true to plot meridional + zonal trendlines
            total: set this to true to plot the mean trendline
            disp_funcs: controls where theoretical functions (e.g., t^3/4 etc.) are displayed
                        the two values in each tuple should be the xmin and xmax
            best_fit: Controls whether a best-fit regression line should be plotted
            cmap: A custom colour map to use when plotting if desired
            norm: Used in conjunction with the cmap var
            '''

    # Set to relative dispersal if a df is given
    if (df is not None) and (ds is None):
        df_bool = True # For further condition passing 
        ds = df.rename(columns={'D2': 'A2', 'Z2': 'disp_x2', 'M2': 'disp_y2', 'ID': 'ID_1', 'ID_marker': 'ID'})
        ds = ds.sort_values('dt')
        df = df.sort_values('dt')
        #df['dt'] = df['dt'] * 3600 * 24
        #ds['dt'] = ds['dt'] * 3600 * 24
        print('Dataframe supplied - attempting to plot relative dispersal...')
        # NOTE: that ds is now a dataframe in this case - I have done this to avoid
        #       using lots of conditional statements given that xarray and pandas
        #       variables are callable in the same way
    elif ds is not None:
        df_bool = False
        
        if cmap is None: # Create a coloUr map if required
            cmap, norm = traj_cmap(ds.traj.values)

    fig = plt.figure(figsize=(4,3), dpi=125)
    ax = fig.add_subplot(1,1,1,)
    # Create the scatter plot if the switch is true
    if scatter:
        if df_bool: # This is the only time conditional statements are required now
                    # since the plot SHOULD function the same now that the df vars
                    # have been given new names to match the ds names
            # Create discrete color map for color bar
            ylab = 'Relative dispersion ($Km^2$)' # Change plot labels and title
            title = 'Relative ' + title
            #cmap = plt.cm.get_cmap("tab20", len(df.ID_marker.unique())) # NOTE: that we also use the new name for ID_marker
            #norm = colors.BoundaryNorm(range(len(df.ID_marker.unique()) + 1), cmap.N)
            
            ax.scatter(
                (df.dt.values) / 3600 / 24, # Very neat because ChatGPT did it
                df.D2.values,
                s=0.5,
                c='gainsboro',
                alpha=0.5
            )
        else:
            # Change labels and titles
            ylab = 'Absoluute dispersion ($Km^2$)' 
            traj_idx, traj_sum = get_traj_index(ds)
            for i, traj in enumerate(ds.ID.values):
                ax.plot(ds.dt[slice(traj_idx[i], traj_sum[i])].values / 3600 / 24, abs(ds.A2[slice(traj_idx[i], traj_sum[i])].values),
                        c='gainsboro', lw=0.5, zorder=0.5, alpha=0.7)
            #scatter = plt.scatter((ds.dt.values)/3600/24, ds.A2.values,
            #                s=0.5, c='gainsboro', alpha=0.5)

    else:
        if df_bool:
            ylab = 'Relative dispersion (Km$^2$)' # Change plot labels and title
        else:
            ylab = 'Absolute dispersion (Km$^2$)' 

    # Find time intervals to average over
    unique_x_values = np.unique((ds.dt.values)/3600/24)
    non_zeros = np.logical_and((unique_x_values > 0), (unique_x_values < max_dt))
    unique_x_values = unique_x_values[non_zeros]
    #print(unique_x_values, len(unique_x_values))

    if total:
        # Calculate the mean dispersion and plot it as solid line
        mean_A2_values = []
        ci_lower = []
        ci_upper = []

        for x_value in unique_x_values:
            values = abs(ds.A2.values[(ds.dt.values)/3600/24 == x_value])
            mean_A2_value = np.nanmean(values)
            mean_A2_values.append(mean_A2_value)

            if ci: # Bootstrap to calculate 95% CI
                n_bootstraps = 1000
                bootstrapped_means = [np.nanmean(np.random.choice(values, size=len(values), replace=True)) for _ in range(n_bootstraps)]
                lower_bound = np.percentile(bootstrapped_means, 5)
                upper_bound = np.percentile(bootstrapped_means, 95)
                ci_lower.append(lower_bound)
                ci_upper.append(upper_bound)

        mean_A2_values = np.array(mean_A2_values)
        ax.plot(unique_x_values, mean_A2_values, color='black', linestyle='solid', label='Total', linewidth=1)

        if ci:
            ci_lower = np.array(ci_lower)
            ci_upper = np.array(ci_upper)
            ax.fill_between(unique_x_values, ci_lower, ci_upper, color='gainsboro', alpha=1, linewidth=0.5)

            #ax.scatter(unique_x_values, mean_plus,
            #        color='gainsboro', alpha=0.5, s=0.5)
            #ax.scatter(unique_x_values, mean_minus,
            #        color='blue', alpha=0.5, s=0.5)

    if components:
        # Calculate mean meridional dispersion at each x value
        mean_x2_values = []
        for x_value in unique_x_values:
            mean_x2_value = np.nanmean(ds.disp_x2.values[(ds.dt.values)/3600/24 == x_value])
            mean_x2_values.append(mean_x2_value)
        ax.plot(unique_x_values, mean_x2_values,
                color='black', linestyle='dashed',
                label='Zonal', linewidth=1, alpha=1)

        # Calculate mean zonal dispersion at each x value
        mean_disp_y2_values = []
        for x_value in unique_x_values:
            mean_disp_y2_value = np.nanmean(ds.disp_y2.values[(ds.dt.values)/3600/24 == x_value])
            mean_disp_y2_values.append(mean_disp_y2_value)
        ax.plot(unique_x_values, mean_disp_y2_values,
                color='black', linestyle='dotted',
                label='Meridional', linewidth=1, alpha=1)
    
    ### POWER LAW FUNCTION ###
    def t_funcs(time, b, a: any = 1000, exp: bool = False):
        ''' Function used to calculate dispersal based on power laws or exponentials.
            t is the time since release, b is the relation to create (e.g., 
            4/3 for RIchardson), and a controls the height on the y axis.
            Change exp to true to use an exponential relation instead of a pwr.'''
        if exp:
            return a * np.exp(b * np.array(time))
        else:
            return a * (np.array(time) ** b)
    
    #def log_formatter(val, pos):
    #    return f'{10**val:.0f}'

    results = [] # Initialise results outside of condition so that function still returns something
    if bool(disp_funcs): # Show theoretical slopes on plot
        for count, f_type in enumerate(disp_funcs):
            if disp_funcs[f_type]['range'][1] != disp_funcs[f_type]['range'][0]: # If the range is the same, then it won't plot
                # Extract desired time range and y values to plot 
                t_vals = [t for t in unique_x_values if disp_funcs[f_type]['range'][0] <= t <= disp_funcs[f_type]['range'][1]]
                t_vals_idx = np.isin((ds.dt.values/3600/24), t_vals)
                t_vals_idx = np.where(t_vals_idx)[0]
                test_t_vals = (ds.dt.values/3600/24)[t_vals_idx]
                test_y_vals = ds.A2.values[t_vals_idx] # Extract dispersion vals

                # Find line of best fit
                print(f_type)

                if ((df_bool) and (f_type == 'exp:f0')): # Automatically add an exponential for relative dispersal
                    reg_results, fitted_vals = fit_trends(x=test_t_vals, y=test_y_vals, plot=False, exp=True)

                    #if theoretical:
                    #    ## Find theoretical distribution
                    #    theoretical_vals = t_funcs(a=reg_results['intercept'][0] + 1000, time=t_vals + disp_funcs[f_type]['range'][0], b=disp_funcs[f_type]['power'], exp=True)

                else:
                    reg_results, fitted_vals = fit_trends(x=test_t_vals, y=test_y_vals, plot=False)

                    #if theoretical:
                    #    # Find theoretical distribution
                    #    theoretical_vals = t_funcs(a=reg_results['intercept'][0] + 1000, time=t_vals  + disp_funcs[f_type]['range'][0], b=disp_funcs[f_type]['power'])

                # Add in regression line        
                if best_fit:
                    #print(f"fitted values {fitted_vals[0], fitted_vals[1]} for func {f_type}")
                    ax.plot(fitted_vals[0], fitted_vals[1],
                        label=f'{f_type}: Best fit dt < {disp_funcs[f_type]["range"][1]}: b={reg_results["slope"].values}',
                        c='steelblue', linestyle='-')

                    mid_index = len(fitted_vals[0]) // 2
                    ax.text(fitted_vals[0][mid_index], fitted_vals[1][mid_index] + 1000, f'm={reg_results["slope"][0]:.2f}',
                            fontsize=9, ha='center', va='bottom')

            if (theoretical) and (f_type != 'exp:f0'):
                # Plot theoretical trendlines
                t = np.linspace(1, 50, 100)
                scale = 0.1 / (1 ** disp_funcs[f_type]['power']) # y-start / 1^pwr law exponent
                theoretical_vals = t_funcs(a=scale, time=t, b=disp_funcs[f_type]['power'])
                plt.plot(t, theoretical_vals,
                         label=f'{f_type}: Theoretical power={disp_funcs[f_type]["power"]}',
                         color='darkgrey', linewidth=1)
                results.append({
                'f_type': f_type,
                'range': disp_funcs[f_type]['range']
                })

                # Add text at the end of each line to indicate the power
                power = disp_funcs[f_type]['power']
                plt.text(t[-1], theoretical_vals[-5], f't$^{power}$', fontsize=9, verticalalignment='bottom', horizontalalignment='right')
   
    # Finish plot and change axese scales, labels, and add legend
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.02, axes_class=plt.Axes)
    ax.tick_params(axis='both', which='major', labelsize=9)
    if log:
        ax.set_yscale('log')
        ax.set_xscale('log')
        # Change log vals to show whole vals
        #ax.xaxis.set_major_formatter(FuncFormatter(log_formatter))
        #ax.yaxis.set_major_formatter(FuncFormatter(log_formatter))
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel(ylab)
    if scatter:
        #cbar = plt.colorbar(mappable=plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax,
        #                            orientation="vertical", ticks=np.arange(len(np.unique(ds.ID.values))) + 0.5)
        #cbar.set_label("ID")
        #unique_ids = np.unique(ds.ID.values)
        #cbar.ax.set_yticklabels(unique_ids)
            
        #cbar.ax.set_yticklabels(np.unique(ds.ID.values)) # Using unique here and above to remove repeated labels
        #                                                 # in the dataframe if it is supplied
        pass

    cax.remove()
    plt.xlim(0, 300)
    plt.ylim(0.005, 50000000)
    plt.show()

    return fig
    #return pd.DataFrame(results)
    #return np.column_stack([test_t_vals, test_y_vals]), np.column_stack([test_t_vals, t_powered])    


def plot_fsle_statistics(fsle_dfs, colors=['r', 'g', 'b', 'c', 'm', 'purple'], index=0, alphas=[1.15, 1.2, 1.25, 1.41, 1.75]):
    """
    Plot the mean FSLE and standard deviation at each separation distance for multiple dataframes.

    Args:
        fsle_dfs (list): A list of pandas.DataFrame containing FSLE values and separation distances.
        colors (list): A list of colors to use for each dataframe. Default is ['r', 'g', 'b', 'c'].

    """
    def bootstrap_ci(data, num_samples=1000, ci=95):
        bootstrapped_means = np.random.choice(data, (num_samples, len(data)), replace=True).mean(axis=1)
        lower_bound = np.percentile(bootstrapped_means, (100-ci)/2)
        upper_bound = np.percentile(bootstrapped_means, 100-(100-ci)/2)
        return lower_bound, upper_bound

    fig = plt.figure(figsize=(6, 6), dpi=125)
    
    for i, fsle_df in enumerate(fsle_dfs):
        # Group by separation distance and calculate nanmean and nanstd
        stats = fsle_df.groupby('separation_distance')['fsle'].agg([np.nanmean]).reset_index()
        stats.rename(columns={'nanmean': 'mean'}, inplace=True)
        stats = stats.sort_values('separation_distance')

        # Scatter plot of all FSLE data points
        #plt.scatter(fsle_df['separation_distance'], fsle_df['fsle'], alpha=0.3, label=f'FSLE Data Points ({i})', color=colors[i])

        # Plot mean FSLE as a line
        if i == index:

            # Initialize lists to store the confidence intervals
            lower_cis = []
            upper_cis = []
            
            # Group by 'separation_distance'
            fsle_grouped = fsle_df.groupby('separation_distance')['fsle']
            
            # Iterate through the groups
            for _, group in fsle_grouped:
                non_nan_group = group.dropna()
                if len(non_nan_group) > 2:
                    lower_ci, upper_ci = bootstrap_ci(non_nan_group)
                else:
                    mean_val = np.nanmean(non_nan_group)
                    lower_ci, upper_ci = mean_val, mean_val
                
                lower_cis.append(lower_ci)
                upper_cis.append(upper_ci)
            
            # Add the confidence intervals to the stats DataFrame
            stats['lower_ci'] = lower_cis
            stats['upper_ci'] = upper_cis

            plt.plot(stats['separation_distance'], stats['mean'],
                     label=r'$\alpha$'+f' = {alphas[i]}',
                     color='black',
                     zorder=9)
            plt.scatter(stats['separation_distance'], stats['mean'],
                        color='black',
                        zorder=10,
                        s=4)

            # Plot standard deviation as points
            #plt.scatter(stats['separation_distance'], abs(stats['mean'] - stats['std']), color=colors[i])
            #plt.scatter(stats['separation_distance'], stats['mean'] + stats['std'], color=colors[i])

            # Plot standard deviation as a shaded area
            plt.fill_between(stats['separation_distance'],
                             stats['lower_ci'],
                             stats['upper_ci'],
                             color='grey',
                             alpha=0.3,
                             zorder=8)
        else:
            plt.plot(stats['separation_distance'], stats['mean'],
                     label=r'$\alpha$'+f' = {alphas[i]}',
                     color=colors[i],
                     alpha=0.8,
                     linewidth=1)
            #plt.scatter(stats['separation_distance'], stats['mean'],
            #            color=colors[i],
            #            s=0.5,
            #            alpha=0.8)

    def t_funcs(time, b, a: any = 1000, exp: bool = False):
        ''' Function used to calculate dispersal based on power laws or exponentials.
            t is the time since release, b is the relation to create (e.g., 
            4/3 for RIchardson), and a controls the height on the y axis.
            Change exp to true to use an exponential relation instead of a pwr.'''
        if exp:
            return a * np.exp(b * np.array(time))
        else:
            return a * (np.array(time) ** b)
    
    # Plot theoretical trendlines
    delta_a = np.linspace(0.06, 2, 20)
    delta_b = np.linspace(2.5, 350, 100)
    delta_c = np.linspace(400, 1000, 100)
    deltas = [delta_a, delta_b, delta_c]
    #deltas = [delta_b, delta_c]
    ystarts = [20, 30, 20000]
    t_powers = [0, -2/3, -2]
    #ystarts = [30, 10000]
    #t_powers = [-2/3, -2]
    for i, ystart in enumerate(ystarts):
        scale = ystart / (1 ** t_powers[i]) # y-start / 1^pwr law exponent
        theoretical_vals = t_funcs(a=scale, time=deltas[i], b=t_powers[i])
        plt.plot(deltas[i], theoretical_vals,
                 color='steelblue', linewidth=1)
        
        # Add text at the end of each line to indicate the power
        labs = ['const', r'$\delta^{\frac{-2}{3}}$', r'$\delta^{-2}$']
        #labs = [r'$\delta^{\frac{-2}{3}}$', r'$\delta^{-2}$']
        lab = labs[i]
        plt.text(deltas[i][-1], theoretical_vals[-5], lab, fontsize=10, verticalalignment='bottom', horizontalalignment='right')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Separation Distance (Km)', fontsize=10)
    plt.ylabel(r'$\lambda$ (days$^{-1}$)', fontsize=10)
    plt.legend(fontsize=10)
    plt.show()

    return fig


def find_chance_pairs(ds, dt: int = 6, ddist = [5, 10]) -> pd.DataFrame:
    ''' Function to find the pairs associated with all drifter trajs
        in a dataset. Takes a separation distance "ddist" (default 10km) and
        time "dt" (default 6h) and returns the ID's, time and points when drifter pairs
        first meet the buffer criteria. Naturaly, SHOULD also find starting pairs.
        
        Args:
            ds: A contiguous ragged array of observations and trajectories of drifters
            dt: The time buffer to use for each point (used to find each time point +-dt)
            ddist: The length of the radial buffer used at each time point'''
    
    def find_first_entry_within_buffer(first_points, second_points, radius_km=[5, 10],
                                       temporal_buffer_hrs=6, full_op: bool=False):
        # Create GeoDataFrames from the first and second point DataFrames
        first_gdf = gpd.GeoDataFrame(first_points, 
                                    crs='EPSG:4326', 
                                    geometry=gpd.points_from_xy(first_points['longitude'], first_points['latitude']))
        #first_gdf = first_gdf.to_crs(crs='EPSG:3112')  # Reproject so that the buffer is more accurate
        #print('***** First GDF *****')
        #print(first_gdf.head())

        second_gdf = gpd.GeoDataFrame(second_points, 
                                    crs='EPSG:4326', 
                                    geometry=gpd.points_from_xy(second_points['longitude'], second_points['latitude']))
        #second_gdf = second_gdf.to_crs(crs='EPSG:3112')
        #print('***** Second GDF *****')
        #print(second_gdf.head())

        # Convert the distance in kilometers to meters
        #radius_m = radius_km * 1000
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=UserWarning)
            r_max = distance.geodesic(kilometers=radius_km[1]).miles / 69
            r_min = distance.geodesic(kilometers=radius_km[0]).miles / 69

            # Create a spatial buffer for the first GeoDataFrame with the maximum radius
            first_gdf['spatial_buffer_max'] = first_gdf['geometry'].buffer(r_max)

            # Create a spatial buffer for the first GeoDataFrame with the minimum radius
            first_gdf['spatial_buffer_min'] = first_gdf['geometry'].buffer(r_min)

            # Create the 'doughnut' shaped mask by taking the difference
            first_gdf['spatial_buffer_combined'] = first_gdf['spatial_buffer_max'].difference(first_gdf['spatial_buffer_min'])

            #Drop intermediate columns
            first_gdf = first_gdf.drop(columns=['geometry','spatial_buffer_max', 'spatial_buffer_min'])

            # Set active geometry to the 'doughnut' mask
            first_gdf = first_gdf.set_geometry(first_gdf['spatial_buffer_combined'], inplace=False)

        # Create a temporal buffer for the first GeoDataFrame
        time_delta = np.timedelta64(temporal_buffer_hrs,'h')
        first_gdf['temporal_buffer_start'] = first_gdf['time'] - time_delta
        first_gdf['temporal_buffer_end'] = first_gdf['time'] + time_delta
        #print('***** Buffer GDF *****')
        #print(first_gdf.head())

        # Spatial filter using sjoin
        CPS_spatial = gpd.sjoin(second_gdf, first_gdf, how='inner',
                                                  predicate='intersects', lsuffix='second', rsuffix='first')
        #print('SB',CPS_spatial.head())

        # Temporal filter using boolean masks
        time_mask = (CPS_spatial['time_second'] >= CPS_spatial['temporal_buffer_start']) & \
                    (CPS_spatial['time_second'] <= CPS_spatial['temporal_buffer_end'])
        #print('TM',time_mask)

        CPS_spatial['ID_marker'] = CPS_spatial['ID_first'].astype(str) + CPS_spatial['ID_second'].astype(str)
        CPS_spatial['ID_marker2'] = CPS_spatial['ID_second'].astype(str) + CPS_spatial['ID_first'].astype(str)

        # Return the first occurrence for each point in the first DataFrame
        if not full_op: # Return only the first point of the pair
            return CPS_spatial[time_mask].drop_duplicates(subset='ID_marker')
        elif full_op: # Return all points that meet the criteria
            return CPS_spatial[time_mask]
    
    # Initialize results dataframe OP DF has 15 columns 
    #results = pd.DataFrame(columns=['index_first','latitude_first','longitude_first','time_first',
    #                                'latitude_second','longitude_second','time_second',
    #                                'geometry','spatial_buffer_max','spatial_buffer_min','spatial_buffer_combined',
    #                                'temporal_buffer_start','temporal_buffer_end',
    #                                'ID_first','ID_second','ID_marker','ID_marker2'])
    results = pd.DataFrame(columns=['ID_marker', 'ID_marker2'])
    # Iterate through all trajs
    for i, first_drifter in enumerate(ds.ID.values):
        print(i,'Run for drifter:', first_drifter)

        # Subset the dataset to include only the provided drifter
        ds_temp1 = retrieve_region(ds, ids=first_drifter, full=False)

        # Use the subset to find time/points of first_drifter
        path1_lat = ds_temp1.latitude.values
        path1_lon = ds_temp1.longitude.values
        path1_times = ds_temp1.time.values
        path1_ids = ds_temp1.ids.values
        path1 = pd.DataFrame({'latitude': path1_lat, 'longitude': path1_lon,
                              'time': path1_times, 'ID': path1_ids})
        #print('***** Path 1 *****')
        #print(path1_lat,len(path1_lat))
        #print(path1_times,len(path1_times))
        #print('Path1 time range', np.datetime64(min(path1['time'])), np.datetime64(max(path1['time'])))
        #print('DS time type:', type(ds.time[0].values), 'e.g.,',ds.time.values[0])

        # Create a second subset to only include times starting from when the first drifter began and ended
        ds_temp2 = retrieve_region(ds, not_ids=first_drifter,
                                   min_time=np.datetime64(min(path1['time'])),
                                   max_time=np.datetime64(max(path1['time'])),full=False)
        #print(len(ds_temp2))
        #print(ds_temp2.time.values)

        #Define index of points in obs dim where each trajectory starts
        traj_idx, trajsum = get_traj_index(ds_temp2)

        # Iterate over the remaining trajectories with higher index values to avoid duplicates
        for j, second_drifter in enumerate(ds_temp2.ID.values):
            #print(j,'Nested run for second_drifter:', second_drifter)

            # Set first ID marker for when results df is empty
            first_ID_marker = (str(second_drifter) + str(first_drifter))
            
            # Save ID of second_drifter and check if it is a duplicate pair
            #print(results['ID_marker'])
            if ((str(first_drifter) + str(second_drifter)) in results['ID_marker'].values) or \
                ((str(first_drifter) + str(second_drifter)) in results['ID_marker2'].values) or \
                ((str(first_drifter) + str(second_drifter)) == first_ID_marker)    :
                print('This msg is here to say that the current pair',first_drifter, second_drifter, 'are not being compared')
                pass
            else:
                # Slice the second subset to go over each drifter in the loop
                slice2 = slice(traj_idx[j], trajsum[j])

                # Use the second slice to find time/points of second_drifter
                path2_times = ds_temp2.time[slice2].values

                path2_lat = ds_temp2.latitude[slice2].values
                path2_lon = ds_temp2.longitude[slice2].values
                path2_ids = ds_temp2.ids[slice2].values
                path2 = pd.DataFrame({'latitude': path2_lat, 'longitude': path2_lon,
                                    'time': path2_times, 'ID': path2_ids})
                #print('***** Path 2 *****')
                #print(path2_lat,len(path2_lat))
                #print(path2_times,len(path2_times))

                # Apply the first_drifter buffer to the second_drifter trajectory
                result = find_first_entry_within_buffer(first_points=path1, second_points=path2,
                                                        radius_km=ddist, temporal_buffer_hrs=dt,
                                                        full_op=False)
                results = pd.concat([results, result], axis=0)
                #if len(result) > 0:
                #    print(results.head())
    return results  


def pairs_plot(df, time: bool = False):
    ''' Plots the first instance of a supplied df of drifter pairs.
        Will also try add in the option to plot full trajectories or
        points at different times for the use in animations etc.'''
    
    cp_fig = plt.figure(dpi=150)
    ax = cp_fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.02, axes_class=plt.Axes)

    # Plot pair positions showing time
    if time:       
        pcm = ax.scatter(df['longitude_first'], df['latitude_first'],
                        s=10, c=pd.to_datetime(df['time_first']).values,
                        cmap=mpl.colormaps['Blues'],
                        transform=ccrs.PlateCarree(), marker='o')
        ax.scatter(df['longitude_second'], df['latitude_second'],
                        s=10, c=pd.to_datetime(df['time_second']).values,
                        cmap=mpl.colormaps['Blues'],
                        transform=ccrs.PlateCarree(), marker='^')
        # Colour bar
        cb = cp_fig.colorbar(pcm, cax=cax)
        cb.set_label('Time (days)')
    else:
        # Create a color map using generated colors
        cmap = sns.color_palette("Set1", n_colors=len(df.ID_marker))
        #cols = [cmap[i-1]for i in df.ID_marker]

        # Plot pair positions       
        pcm = ax.scatter(df['longitude_first'], df['latitude_first'],
                        s=10, cmap=cmap, c=df['traj_cols_first'],
                        transform=ccrs.PlateCarree(), marker='o')
        ax.scatter(df['longitude_second'], df['latitude_second'],
                        s=10, cmap=cmap, c=df['traj_cols_second'],
                        transform=ccrs.PlateCarree(), marker='^')
        # Colour bar
        cb = cp_fig.colorbar(pcm, cax=cax)
        cb.set_label('Pair Number')
                
        #cb.ax.set_yticklabels(np.unique(df.ID_marker.values))


    # Add borders
    ax.add_feature(cfeature.LAND, facecolor='grey', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.25, zorder=1)

    # Set domain
    ax.set_extent((140,175,-50,-15), crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(140,175,5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-50,-15,5), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())


def pair_driftplot(df_list, ds: any = None, labels: list = ['20-40 km', '10-20 km', '5-10 km', '0-5 km'],
                   symbols: list = ['o', 's', 'v', 'P'], ls: list = ['-', '--', '-.', ':'],
                   alphas = [0.6, 0.7, 0.8, 0.9], domain = [145, 165, -45, -20]) -> plt.figure:
    """
    Function to plot drifter pairs on a map.

    Args:
        df_list (list): List of pandas dataframes containing drifter pairs data.
        ds (any, optional): Drifter dataset to plot trajectories from. Defaults to None.
        labels (list, optional): Labels for the legend. Defaults to ['20-40 km', '10-20 km', '5-10 km', '0-5 km'].
        symbols (list, optional): Marker symbols for the legend. Defaults to ['o', 's', 'v', 'P'].
        ls (list, optional): Line styles for the legend. Defaults to ['-', '--', '-.', ':'].
        alphas (list, optional): Alpha values for the plot. Defaults to [0.6, 0.7, 0.8, 0.9].
        domain (any, optional): Domain to set for the plot. Defaults to None.

    Returns:
        plt.figure: Matplotlib figure object.
    """

    # Create figure 
    fig = plt.figure(figsize=(6,6), dpi=125)
    ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())

    # Plot trajectories from the drifter dataset
    if ds is not None:
        traj_idx, traj_sum = get_traj_index(ds)
        for i, traj in enumerate(ds.ID.values):
            ax.plot(ds.longitude[slice(traj_idx[i], traj_sum[i])], ds.latitude[slice(traj_idx[i], traj_sum[i])],
                    c='gainsboro', lw=0.5, zorder=0.5, alpha=0.7)

    # Generate a colormap and list of colors
    colormap = cmaps.cet_g_bw
    n_colors = 256  # Number of distinct colors you want to use
    cols = [colormap(i / n_colors) for i in range(n_colors)]

    # Plot drifter pairs
    for i, df in enumerate(df_list):
        df_grouped = df.groupby('ID_marker')
        for j, (group_id, group) in enumerate(df_grouped):
            alpha = alphas[i]
            color = cols[j % len(cols)]  # Assign color cyclically if more groups than colors
            ax.plot(group['longitude_first'], group['latitude_first'], zorder=0.6, c=color, lw=1,
                    ls=ls[i % len(ls)], alpha=alpha)
            ax.plot(group['longitude_second'], group['latitude_second'], zorder=0.7, c=color, lw=1,
                    ls=ls[i % len(ls)], alpha=alpha)
            ax.scatter(group['longitude_first'].iloc[0], group['latitude_first'].iloc[0], zorder=0.8, 
                       edgecolors='black', facecolor=color, marker=symbols[i % len(symbols)], s=20,
                       alpha=alpha)
            ax.scatter(group['longitude_second'].iloc[0], group['latitude_second'].iloc[0], zorder=0.8, 
                       edgecolors='black', facecolor=color, marker=symbols[i % len(symbols)], s=20,
                       alpha=alpha)

    # Add land and coastlines to plot 
    ax.add_feature(cfeature.LAND, facecolor='silver', zorder=0.9)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=1)

    # Set the displayed region
    ax.set_extent(domain, crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(domain[0],domain[1],5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(domain[2],domain[3],5), crs=ccrs.PlateCarree())
    
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    # Create proxy artists for the legend
    proxy_artists = [plt.Line2D([0], [0], marker=symbol, color='black', markerfacecolor='none', markersize=10, linestyle='None')
                     for symbol in symbols]

    # Add legend to the plot without background or border
    ax.legend(proxy_artists, labels, loc='center left', bbox_to_anchor=(0, 0.75), frameon=False)

    # Add inset map of Australia
    inset_ax = fig.add_axes([0.22, 0.455, 0.19, 0.19], projection=ccrs.PlateCarree())
    inset_ax.set_extent([110, 168, -48, -10], crs=ccrs.PlateCarree())
    inset_ax.add_feature(cfeature.LAND, facecolor='silver')
    inset_ax.add_feature(cfeature.COASTLINE, lw=0.5)

    # Plot the 'domain' as a black box on the inset map
    inset_ax.plot([domain[0], domain[1], domain[1], domain[0], domain[0]], 
                  [domain[2], domain[2], domain[3], domain[3], domain[2]], 
                  c='black', transform=ccrs.PlateCarree(), lw=0.5)

    return fig


def rel_disp(ds) -> pd.DataFrame:
    ''' Calculates the separation distance between all drifter pairs and calculates the
        relative dispersion. NOTE: this function finds ALL pairs in time, so it can be 
        quite slow if given a large amount of data.
         '''
    
    def find_active_trajs(ds):
        ''' Takes a drifter dataset and finds the trajectories that occur at the same time
        '''

        start_indices, startsums = get_traj_index(ds)
        num_arrays = len(start_indices)

        shared_indices = []
        
        for i in range(num_arrays - 1):
            for j in range(i + 1, num_arrays):
                # Get the start and end indices for the two arrays
                start_idx1, start_idx2 = start_indices[i], start_indices[j]
                end_idx1 = start_indices[i + 1] if i < num_arrays - 1 else len(ds.time.values)
                end_idx2 = start_indices[j + 1] if j < num_arrays - 1 else len(ds.time.values)

                # Extract the time values for the two arrays
                time1 = ds.time[start_idx1:end_idx1].values
                time2 = ds.time[start_idx2:end_idx2].values

                # Find the intersection of time values between the two arrays
                intersection = list(set(time1) & set(time2))

                if intersection:
                    shared_indices.append((ds.ID[i].values, ds.ID[j].values))

        # Convert the result to a pandas DataFrame
        result_df = pd.DataFrame(shared_indices, columns=['ID1', 'ID2'])

        return result_df, shared_indices
    
    def distance_series(ds, start_sep: any = None, ids: any = None):
        ''' Calculates the distance between two drifters for all times they are active together
            Basically a nested, neater version of the 'pair_sep' function (but here we also use
            mins instead of maxs for algning times)
        '''
        
        # Subset to get trajectory of drifters
        ds_temp1 = retrieve_region(ds, ids=ids[0])
        ds_temp2 = retrieve_region(ds, ids=ids[1])

        # Extract time values from each subset
        times1 = ds_temp1.time.values
        times2 = ds_temp2.time.values

        # Find and align times
        aligned_times1, aligned_times2, temp1_indices, temp2_indices = min_max_align(times1, times2, use_min=True)

        # Create list of dts based on aligned time values
        dt = (aligned_times1 - min(aligned_times1)).astype('timedelta64[s]').view('int64')

        # Extract lat/lon at aligned indices
        lat_temp1 = ds_temp1.latitude[temp1_indices].values
        lon_temp1 = ds_temp1.longitude[temp1_indices].values
        lat_temp2 = ds_temp2.latitude[temp2_indices].values
        lon_temp2 = ds_temp2.longitude[temp2_indices].values

        # Create a matrix of latitude and longitude pairs
        d1 = np.column_stack((lat_temp1, lon_temp1))
        d2 = np.column_stack((lat_temp2, lon_temp2))
    
        # Initialize distance arrays
        #distances = np.full(len(temp1_indices), fill_value=np.nan, dtype=float)
        #dist_zonal = np.full(len(temp1_indices), fill_value=np.nan, dtype=float)
        #dist_meridional = np.full(len(temp1_indices), fill_value=np.nan, dtype=float)
        distances = []
        dist_zonal = []
        dist_meridional = []

        # Calculate distances using matrix operations
        for i in range(len(aligned_times1)):
            dist = distance.distance(d1[i], d2[i]).km
            zonal_i = distance.distance(d1[i],(d2[i][0],d1[i][1])).km
            meridional_i = distance.distance(d1[i],(d1[i][0],d2[i][1])).km
            #distances[i] = dist
            #dist_zonal[i] = zonal_i
            #dist_meridional[i] = meridional_i
            distances = np.append(distances, dist)
            dist_zonal = np.append(dist_zonal, zonal_i)
            dist_meridional = np.append(dist_meridional, meridional_i)

        # Calculate D^2 for relative dispersion
        D2 = distances ** 2
        Z2 = dist_zonal ** 2
        M2 = dist_meridional ** 2

        # Create new unique ID for each pair
        ids_1 = np.full_like(distances, fill_value=ds_temp1.ID[0].values, dtype=np.int64)
        ids_2 = np.full_like(distances, fill_value=ds_temp2.ID[0].values, dtype=np.int64)
        ids_1 = ids_1.astype(str)
        ids_2 = ids_2.astype(str)
        ID_marker = [str1 + str2 for str1, str2 in zip(ids_1, ids_2)]
        traj_cols_first = np.full_like(distances, fill_value=np.nan)
        traj_cols_second = np.full_like(distances, fill_value=np.nan) # NOTE: left empty bcoz pandas doesnt like multiple values

        # BUG CHECKING
        #print('ids1:', len(ids_1))
        #print('ids2:', len(ids_2))
        #print('ID_marker:', len(ID_marker))
        #print('Time1:', len(aligned_times1))
        #print('Time2:', len(aligned_times2))
        #print('dt:', len(dt))
        #print('distances:', len(distances))
        #print('traj_c1:', len(traj_cols_first))
        #print('traj_c2:', len(traj_cols_second))
        #print('D2:', len(D2))

        # Create a DataFrame with distances and time values
        df_dist = pd.DataFrame(data={'ID': ids_1, 'comp_ID': ids_2, 'ID_marker': ID_marker, 'time_first': aligned_times1,
                                     'time_second': aligned_times2, 'dt': dt, 'sep_distance': distances, 'traj_cols_first': traj_cols_first,
                                     'traj_cols_second': traj_cols_second, 'D2': D2, 'dist_zonal': dist_zonal,
                                     'dist_meridional': dist_meridional, 'Z2': Z2, 'M2': M2},
                            columns=['ID','comp_ID', 'ID_marker', 'time_first', 'time_second', 'dt', 'sep_distance', 'traj_cols_first',
                                     'traj_cols_second', 'D2', 'dist_zonal', 'dist_meridional', 'Z2', 'M2'])
        return df_dist
    
    # Find active trajectory matches
    actives_df, actives_pairs = find_active_trajs(ds)

    # find separation time series for each match
    separations = pd.DataFrame()
    for i in tqdm(range(len(actives_df))):
        ID1 = actives_pairs[i][0]
        ID2 = actives_pairs[i][1]
        separation = distance_series(ds=ds, ids=[ID1, ID2])

        #sep_distances = separation['sep_distance'].values
        #delta_times = separation['dt'].values

        separations = pd.concat([separations, separation], ignore_index=True)

    return separations


def compute_fsle(df, threshold_rate=1.2, initial_distance=0.01):
    """
    Compute the Finite Scale Lyapunov Exponent (FSLE) from a dataframe containing particle pair separation distances and time lags.

    Args:
        df (pandas.DataFrame): Dataframe containing particle pair information with columns ['ID_marker', 'sep_distance', 'dt'].
        threshold_rate (float, optional): Threshold rate `r` used to define the series of thresholds `d_n = r^n * d_0`. Default is 2.0.
        initial_distance (float, optional): Initial separation distance `d_0`. Default is 0.01.

    Returns:
        pandas.DataFrame: Dataframe containing the FSLE values for different separation distances.
    """
    thresholds = initial_distance * threshold_rate ** np.arange(100)  # Define a series of thresholds

    fsle_data = []

    for pair_id in df['ID_marker'].unique():
        pair_df = df[df['ID_marker'] == pair_id]
        pair_df = pair_df.sort_values('dt')  # Sort by time lag
        pair_df = pair_df.reset_index(drop=True)  # Reset index

        for i, threshold in enumerate(thresholds):
            if pair_df['sep_distance'].max() < threshold:
                continue  # Skip if maximum separation is less than the current threshold

            # Find the first instance where separation distance exceeds the threshold
            idx = np.argmax(pair_df['sep_distance'] >= threshold)
            if idx == 0 or pair_df['sep_distance'].iloc[idx - 1] >= threshold:
                fsle_data.append({'ID_marker': pair_id, 'separation_distance': threshold, 'fsle': np.nan})
                continue

            init_dist = pair_df['sep_distance'].iloc[idx - 1]
            final_dist = pair_df['sep_distance'].iloc[idx]
            time_lag = pair_df['dt'].iloc[idx] - pair_df['dt'].iloc[idx - 1]

            if time_lag == 0:
                fsle = np.nan
            else:
                # Compute FSLE using ln(r) = ln(final_dist / init_dist) / time_lag
                fsle = np.log(final_dist / init_dist) / time_lag

            fsle_data.append({'ID_marker': pair_id, 'separation_distance': threshold, 'fsle': fsle})

    fsle_df = pd.DataFrame(fsle_data)

    def filter_separations(separations):

        def has_time_info(date_str):
            try:
                pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S')
                return True
            except ValueError:
                return False

        separations['dt'] = separations['dt'] / 3600 / 24
        # Filter out rows where 'time_first' or 'time_second' do not include time information
        df = separations[separations['time_first'].apply(has_time_info) & separations['time_second'].apply(has_time_info)]

        # Check if 'time_first' is not datetime and convert if necessary
        if not np.issubdtype(df['time_first'].dtype, np.datetime64):
            df['time_first'] = pd.to_datetime(df['time_first'])

        # Check if 'time_second' is not datetime and convert if necessary
        if not np.issubdtype(df['time_second'].dtype, np.datetime64):
            df['time_second'] = pd.to_datetime(df['time_second'])

        # Calculate the time difference
        df['time_diff'] = (df['time_second'] - df['time_first']).abs()

        # Filter rows where the time difference is greater than 1 hour
        filtered_df = df[df['time_diff'] <= pd.Timedelta(hours=1)]

        # Drop the 'time_diff' column if it's no longer needed
        filtered_df = filtered_df.drop(columns=['time_diff'])

        print(len(filtered_df))
        print(len(separations))
        return filtered_df

    return filter_separations(fsle_df)


def calculate_cist(df, r_thresh=0.5, initial_distance=0.01, alpha=1.2):
    """
    Calculate the cumulative inverse separation time (CIST) from a dataframe of particle pair data.

    Args:
        df (pandas.DataFrame): Dataframe containing columns 'ID', 'distance', and 'dt'.
        r_thresh (float, optional): Threshold value for the cumulative density function (CDF). Default is 0.5.
        initial_distance (float, optional): Initial separation distance for calculating the bins. Default is 0.01.
        alpha (float, optional): Ratio between subsequent separation distance bins (r_bins[i+1] = alpha * r_bins[i]). Default is 1.2.

    Returns:
        pandas.DataFrame: Dataframe containing columns 'ID_marker', 'r_bin' (separation distance bins), and 'cist' (cumulative inverse separation time).
    """
    # Group the data by particle pair ID
    grouped = df.groupby('ID')

    # Initialize lists to store CIST values, separation distances, and IDs
    r_bins = initial_distance * alpha ** np.arange(100)  # Define a series of thresholds
    cist_values = []
    r_values = []
    id_values = []

    for r_bin in r_bins:
        # Filter the data for pairs that reach the current separation distance bin
        filtered = grouped.filter(lambda x: x['sep_distance'].max() >= r_bin)

        if not filtered.empty:
            # Calculate the times when the CDF crosses the threshold for each pair
            crossing_times = filtered.groupby('ID_marker').apply(lambda x: x.loc[x['sep_distance'].cumsum() / x['sep_distance'].sum() >= r_thresh, 'dt'].values[0])

            # Calculate the CIST as the inverse of the mean time difference between the current bin and the previous bin
            if r_bin == r_bins[0]:
                cist_values.extend(1 / crossing_times)
                r_values.extend([r_bin] * len(crossing_times))
                id_values.extend(filtered['ID_marker'].unique())
            else:
                prev_crossing_times = filtered.groupby('ID_marker').apply(lambda x: x.loc[x['sep_distance'].cumsum() / x['sep_distance'].sum() >= r_thresh / alpha, 'dt'].values[0])
                cist_values.extend(1 / (crossing_times - prev_crossing_times))
                r_values.extend([r_bin] * len(crossing_times))
                id_values.extend(filtered['ID_marker'].unique())

        #else:
        #    cist_values.append(np.nan)
        #    r_values.append(r_bin)
        #    id_values.append(np.nan)

        #print(len(id_values), len(r_values), len(cist_values))
        #print(id_values)
        #print(r_values)
        #print(cist_values)

    return pd.DataFrame({'ID_marker': id_values, 'r_bin': r_values, 'cist': cist_values})


def calculate_triangle_pars(lats1, lons1, lats2, lons2, lats3, lons3) -> pd.DataFrame:
    '''Calculates time series of area, scaled area vs perimeter, and max angle
        for drifter triads.'''

    def triangle_pars(lat1, lon1, lat2, lon2, lat3, lon3):
        # Check if all drifters are active
        # if any drifter isn't active, return NA
        if any(np.isnan([lat1, lon1, lat2, lon2, lat3, lon3])):
            triangle_area = np.nan
            lambda_ratio = np.nan
            max_angle = np.nan
        else:
            # Calculate side lengths using semi-spherical geometry
            a = distance.geodesic((lat2, lon2), (lat1, lon1)).meters
            b = distance.geodesic((lat3, lon3), (lat2, lon2)).meters
            c = distance.geodesic((lat1, lon1), (lat3, lon3)).meters

            # Calculate perimeter and semiperimeter
            p = a + b + c
            s = p / 2

            # Calculate area
            triangle_area = np.sqrt(s * (s - a) * (s - b) * (s - c))

            # Calculate lambda ratio
            lambda_ratio = 12 * np.sqrt(3) * (triangle_area / p**2)

            # Calculate angles and find the max NOTE: it might be worthwhile looking into accounting for the semi-spherical geometry for angles too
            # I think this should be accounted for already though, since we calculate distances using a geodesic
            angles = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)), np.arccos((c**2 + a**2 - b**2) / (2 * c * a)), np.arccos((a**2 + b**2 - c**2) / (2 * a * b))

            max_angle = np.degrees(np.max(angles))
        return triangle_area, lambda_ratio, max_angle

    # Only run up to the length of the smallest trajectory
    min_traj_len = min(len(lats1), len(lats2), len(lats3))

    # Determine parameters at each time NOTE: that it is assumed arrays are aligned in time
    areas = []
    max_angles = []
    lambda_ratios = []
    for i in range(min_traj_len):
        lat1 = lats1[i]
        lon1 = lons1[i]
        lat2 = lats2[i]
        lon2 = lons2[i]
        lat3 = lats3[i]
        lon3 = lons3[i]
        area, max_angle, lambda_ratio = triangle_pars(lat1, lon1, lat2, lon2, lat3, lon3)
        
        areas.append(area)
        max_angles.append(max_angle)
        lambda_ratios.append(lambda_ratio)

    result_df = pd.DataFrame({
        'Triangle Area': areas,
        'Max Interior Angle': max_angles,
        'Lambda Ratio': lambda_ratios
    })

    return result_df


def triangles_plot(points_series, domain: list = [145,165,-45,-20], borders: bool = True) -> plt.figure:
    ''' Function to plot a series of triangles formed by drifters onto a map'''

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Set the map extent based on the specified region
    ax.set_extent(domain, crs=ccrs.PlateCarree())

    # Add borders and features
    if borders:
        ax.add_feature(cfeature.LAND, facecolor='grey', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.25, zorder=1)

    for points in points_series:
        lats, lons = points
        ax.plot(lons + [lons[0]], lats + [lats[0]],
                marker='o', markersize=5,
                transform=ccrs.PlateCarree(), label='Drifters'
                )
        
        for i in range(len(lats)): 
            triangle = plt.Polygon([(lons[i], lats[i]),
                                    (lons[(i + 1) % len(lats)], lats[(i + 1) % len(lats)]),
                                    (lons[(i + 2) % len(lats)], lats[(i + 2) % len(lats)])],
                                    edgecolor='black', facecolor='blue', transform=ccrs.PlateCarree()
                                    )
            ax.add_patch(triangle)
    
    ax.legend()
    ax.set_title('Drifter Triad Time Series')
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    plt.show()

