# Drifter triad functions #

# lIBRARY #
# -----------------------------------------------------------------------

# Data manipulation #
import numpy as np
import xarray as xr
import pandas as pd

# Others #
from datetime import datetime
import os

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

def find_first_indices(arr):
    ''' Used to find where each trajectory in the data starts'''
    unique_values, indices = np.unique(arr, return_index=True)
    return unique_values, indices

def triangle_pars(lat1, lon1, lat2, lon2, lat3, lon3):
    # Check if all drifters are active
    # if any drifter isn't active, return NA
    if sum(np.isnan([lat1, lon1, lat2, lon2, lat3, lon3])[0]) > 0:
        triangle_area = np.nan
        lambda_ratio = np.nan
        max_angle = np.nan
    else:
        # Calculate side lengths using semi-spherical geometry
        coordinates = np.array([
            [lat1, lon1, lat2, lon2],
            [lat1, lon1, lat3, lon3],
            [lat2, lon2, lat3, lon3]
        ])

        # Calculate distances for all sides using vectorized operations
        distances = np.apply_along_axis(lambda x: distance.geodesic((x[0], x[1]), (x[2], x[3])).m, axis=1, arr=coordinates)

        # Assign the distances to variables a, b, and c
        a, b, c = distances
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

def lambda_qc(df, threshold = 0.2):
    ''' Filters out triangle parameters from a dataframe that are outside of a specified threshold
        of lambda ratios (see the triangle pars function for calculation of the lambda ratio)'''
    
    return df[df['lambda_ratio'] > threshold]

def plot_triangles(lat_lon_arrays, domain: list = [151.5, 153.3, -33.8, -32.2], borders: bool = True,
                   cvec: any = 'blue', bathy: any = None):
    ''' Function to plot triangles formed by drifters onto a map'''

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Set the map extent based on the specified region
    ax.set_extent(domain, crs=ccrs.PlateCarree())

    # Add borders and features
    if borders:
        ax.add_feature(cfeature.LAND, facecolor='grey', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.25, zorder=1)

        latBathy = bathy.variables['lat'][:]       
        lonBathy = bathy.variables['lon'][:]       
        heightBathy = bathy.variables['height'][:]
        ax.contour(lonBathy, latBathy, heightBathy,
                    levels = [-2000, -1000, -200], linewidths=1,
                    colors = 'grey', linestyles='solid', zorder=0.2)
    
    # Create colour map and apply to 'cvec'
    cmap = cmocean.cm.speed
    norm = colors.Normalize(vmin=-3, vmax=3)

    alpha = 0.65  # Initial alpha value
    for i in range(0, len(lat_lon_arrays[0]), 3):
        lats = [lat_lon_array[i] for lat_lon_array in lat_lon_arrays[::2]]
        lons = [lat_lon_array[i] for lat_lon_array in lat_lon_arrays[1::2]]

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
    cbar = plt.colorbar(sm, ax=ax, label='Triangle Area')

    # Set the displayed region
    if domain:
        ax.set_extent(domain, crs=ccrs.PlateCarree())
        ax.set_xticks(np.arange(domain[0],domain[1],0.2), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(domain[2],domain[3],0.2), crs=ccrs.PlateCarree())

    ax.set_title('Drifter Triad Time Series')
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    return fig


def animate_triangles(lat_lon_arrays, domain: list = [151.5, 153.3, -33.8, -32.2],
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
            ax.set_xticks(np.arange(domain[0],domain[1],0.2), crs=ccrs.PlateCarree())
            ax.set_yticks(np.arange(domain[2],domain[3],0.2), crs=ccrs.PlateCarree())

        ax.set_title(f'Drifter Triad at Time {frame}')
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())

    anim = FuncAnimation(fig, update, frames=len(lat_lon_arrays[0]), repeat=False, blit=False)

    # Save the animation as a GIF
    anim.save(save_path, writer='pillow', fps=framerate)


