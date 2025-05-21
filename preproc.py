#-------------------------------------------------------------------------------------------------------------------------#
''' Code to perform interpolation and convert data into ragged arrays'''

import xarray as xr
import numpy as np
from datetime import datetime
import pandas as pd
import awkward as ak
from tqdm import tqdm

def fill_values(var, default=np.nan):
    '''
    Change fill values (-1e+34, inf, -inf) in var array to value specified by default
    '''
    missing_value = np.logical_or(np.isclose(var, -1e+34), ~np.isfinite(var))
    if np.any(missing_value):
        var[missing_value] = default
    return var


def str_to_float(value, default=np.nan):
    '''
    :param value: string
    :return: bool
    '''
    try:
        fvalue = float(value)
        if np.isnan(fvalue):
            return default
        else:
            return fvalue
    except ValueError:
        return default

def cut_str(value, max_length):
    '''
    Cut a string to a specify lenth.
    :param value: string
           max_length: lenght of the output
    :return: string with max_length chars
    '''
    return value[:max_length]

class create_ragged_arr:
    def __init__(self, records):
        self.records = records # Records should be given as a list of xr dicts
        self.rowsize = self.number_of_observations(self.records)
        self.nb_traj = len(self.rowsize)
        self.nb_obs = np.sum(self.rowsize).astype('int')
        self.allocate_data(self.nb_traj, self.nb_obs)
        self.index_traj = np.insert(np.cumsum(self.rowsize), 0, 0)

        for i, ds in tqdm(enumerate(self.records), total=len(self.records)):
            self.fill_ragged_array(ds, i, self.index_traj[i])

    def number_of_observations(self, records) -> np.array:
        '''
        Load records and get the size of the observations.
        '''
        rowsize = np.zeros(len(records), dtype='int')
        for i, record in enumerate(records):
            ds = xr.Dataset(record)
            rowsize[i] = ds.sizes['obs']
        return rowsize
    
    def allocate_data(self, nb_traj, nb_obs):
        '''
        Reserve the space for the total size of the array.
        Most fields are left commented out since these are
        dropped eventually. If they want to be included in
        the final ds, just uncomment.
        '''
        # Metadata -> uncomment as needed
        self.id = np.zeros(nb_traj, dtype='<U20')
        self.deploy_lat = np.zeros(nb_traj, dtype='float32')
        self.deploy_lon = np.zeros(nb_traj, dtype='float32')
        
        # values defined at every observation (timestep)
        self.longitude = np.zeros(nb_obs, dtype='float32')
        self.latitude = np.zeros(nb_obs, dtype='float32')
        self.time = np.zeros(nb_obs, dtype='datetime64[s]')
        self.dt_diff = np.zeros(nb_obs, dtype='int64')
        self.ve = np.zeros(nb_obs, dtype='float32')
        self.vn = np.zeros(nb_obs, dtype='float32')
        self.hpa = np.zeros(nb_obs, dtype='float32')
        self.pt_d = np.zeros(nb_obs, dtype='float32')
        self.pt_x = np.zeros(nb_obs, dtype='float32')
        self.pt_y = np.zeros(nb_obs, dtype='float32')
        self.latres = np.zeros(nb_obs, dtype='float32')
        self.lonres = np.zeros(nb_obs, dtype='float32')
        self.u_res = np.zeros(nb_obs, dtype='float32')
        self.v_res = np.zeros(nb_obs, dtype='float32')
        
        # sst data set
        self.sst = np.zeros(nb_obs, dtype='float32')

        # Signal paramaters
        self.div = np.zeros(nb_obs, dtype='float32')
        self.Ro = np.zeros(nb_obs, dtype='float32')
        self.R = np.zeros(nb_obs, dtype='float32')
        self.V = np.zeros(nb_obs, dtype='float32')
        self.omega = np.zeros(nb_obs, dtype='float32')
        self.kappa = np.zeros(nb_obs, dtype='float32')
        self.lambda_val = np.zeros(nb_obs, dtype='float32')
        self.theta = np.zeros(nb_obs, dtype='float32')
        self.phi = np.zeros(nb_obs, dtype='float32')
        self.psi = np.zeros(nb_obs, dtype='float32')
        self.zo = np.zeros(nb_obs, dtype='complex128')

    def fill_ragged_array(self, record, tid, oid):
        '''
        Fill the ragged array from the xr.Dataset() corresponding to one trajectory

        Input filename: path and filename of the netCDF file
              tid: trajectory index
              oid: observation index in the ragged array
        '''
        ds = xr.Dataset(record) # This assumes that data is given as a dictionary
        size = ds.dims['obs']

        # scalar
        self.id[tid] = str(ds.ID.data[0])
        self.deploy_lon[tid] = ds.deploy_lon.data[0]
        self.deploy_lat[tid] = ds.deploy_lat.data[0]
        
        # vectors
        self.longitude[oid:oid+size] = ds.longitude.data[0]
        self.latitude[oid:oid+size] = ds.latitude.data[0]
        self.time[oid:oid+size] = (ds.time.data[0])
        try: self.dt_diff[oid:oid+size] = ds.dt_diff.data[0]
        except: pass
        try: self.ve[oid:oid+size] = ds.ve.data[0]
        except: pass
        try: self.vn[oid:oid+size] = ds.vn.data[0]
        except: pass
        try: self.sst[oid:oid+size] = (ds.sst.data[0])
        except: pass
        try: self.hpa[oid:oid+size] = ds.hpa.data[0]
        except: pass
        try: self.pt_d[oid:oid+size] = ds.pt_d.data[0]
        except: pass
        try: self.pt_x[oid:oid+size] = ds.pt_x.data[0]
        except: pass
        try: self.pt_y[oid:oid+size] = ds.pt_y.data[0]
        except: pass
        try: self.latres[oid:oid+size] = ds.latres.data[0]
        except: pass
        try: self.lonres[oid:oid+size] = ds.lonres.data[0]
        except: pass
        try: self.u_res[oid:oid+size] = ds.u_res.data[0]
        except: pass
        try: self.v_res[oid:oid+size] = ds.v_res.data[0]
        except: pass

        # Signal paramaters
        try: self.div[oid:oid+size] = ds.div.data[0]
        except: pass
        try: self.Ro[oid:oid+size] = ds.Ro.data[0]
        except: pass
        try: self.R[oid:oid+size] = ds.R.data[0]
        except: pass
        try: self.V[oid:oid+size] = ds.V.data[0]
        except: pass
        try: self.omega[oid:oid+size] = ds.omega.data[0]
        except: pass
        try: self.kappa[oid:oid+size] = ds.kappa.data[0]
        except: pass
        try: self.lambda_val[oid:oid+size] = ds['lambda_val'].data[0]
        except: pass
        try: self.theta[oid:oid+size] = ds.theta.data[0]
        except: pass
        try: self.phi[oid:oid+size] = ds.phi.data[0]
        except: pass
        try: self.psi[oid:oid+size] = ds.psi.data[0]
        except: pass
        try: self.zo[oid:oid+size] = ds.zo.data[0]
        except: pass

    def to_xarray(self):
        ds = xr.Dataset(
            data_vars=dict(
                rowsize=(['traj'], self.rowsize, {'long_name': 'Number of observations per trajectory', 'sample_dimension': 'obs', 'units':'-'}),
                deploy_lon=(['traj'], self.deploy_lon, {'long_name': 'Deployment longitude', 'units':'degrees_east'}),
                deploy_lat=(['traj'], self.deploy_lat, {'long_name': 'Deployment latitude', 'units':'degrees_north'}),
                
                # position and velocity
                dt_diff=(['obs'], self.dt_diff, {'long_name': 'Time interval between previous and next location', 'units':'s'}),
                ve=(['obs'], self.ve, {'long_name': 'Eastward velocity', 'units':'m/s'}),
                vn=(['obs'], self.vn, {'long_name': 'Northward velocity', 'units':'m/s'}),
                latres=(['obs'], self.latres, {'long_name': 'Latitude resolution', 'units':'degrees_north'}),
                lonres=(['obs'], self.lonres, {'long_name': 'Longitude resolution', 'units':'degrees_east'}),
                u_res=(['obs'], self.u_res, {'long_name': 'Eastward residual velocity', 'units':'m/s'}),
                v_res=(['obs'], self.v_res, {'long_name': 'Northward residual velocity', 'units':'m/s'}),
                pt_d=(['obs'], self.pt_d, {'long_name': 'Distance to previous location', 'units':'m'}),
                pt_x=(['obs'], self.pt_x, {'long_name': 'Distance to previous location in x', 'units':'m'}),
                pt_y=(['obs'], self.pt_y, {'long_name': 'Distance to previous location in y', 'units':'m'}),
                
                # sst hpa
                sst=(['obs'], self.sst, {'long_name': 'Fitted sea water temperature', 'units':'Kelvin', 'comments': 'Estimated near-surface sea water temperature from drifting buoy measurements. It is the sum of the fitted near-surface non-diurnal sea water temperature and fitted diurnal sea water temperature anomaly. Discrepancies may occur because of rounding.'}),
                hpa=(['obs'], self.hpa, {'long_name': 'Fitted sea-level air pressure', 'units':'hPa', 'comments': 'Sea-level air pressure from drifting buoy measurements.'}),
                div=(['obs'], self.div, {'long_name': 'Normalized Divergence', 'units':'-'}),
                Ro=(['obs'], self.Ro, {'long_name': 'Rossby number', 'units':'-'}),
                R=(['obs'], self.R, {'long_name': 'Eddy Radius', 'units':'Km'}),
                V=(['obs'], self.V, {'long_name': 'Eddy Velocity', 'units':'m/s'}),
                omega=(['obs'], self.omega, {'long_name': 'Non-dimensional instantaneous frequency', 'units':'-'}),
                kappa=(['obs'], self.kappa, {'long_name': 'Ellipse amplitude', 'units':'Km'}),
                lambda_val=(['obs'], self.lambda_val, {'long_name': 'Ellipse linearity', 'units':'-'}),
                theta=(['obs'], self.theta, {'long_name': 'Ellipse orientation', 'units':'Radians'}),
                phi=(['obs'], self.phi, {'long_name': 'Ellipse phase', 'units':'-'}),
                psi=(['obs'], self.psi, {'long_name': 'Particle position in ellipse', 'units':'-'}),
                zo=(['obs'], self.zo, {'long_name': 'Ellipse center', 'units':'-'}),
             ),

            coords=dict(
                ID=(['traj'], self.id, {'long_name': 'Global Drifter Program Buoy ID', 'units':'-'}),
                longitude=(['obs'], self.longitude, {'long_name': 'Longitude', 'units':'degrees_east'}),
                latitude=(['obs'], self.latitude, {'long_name': 'Latitude', 'units':'degrees_north'}),
                time=(['obs'], self.time, {'long_name': 'Time'}),
                ids=(['obs'], np.repeat(self.id, self.rowsize), {'long_name': "Global Drifter Program Buoy identification number repeated along observations", 'units':'-'}),
            ),

            attrs={
                'title': 'Global Drifter Program hourly drifting buoy collection',
                'date_created': datetime.now().isoformat(),
            }
        )

        #ds.time.encoding['units'] = 'seconds since 1970-01-01 00:00:00'

        return ds
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

from scipy.interpolate import interp1d, LinearNDInterpolator, griddata
from pykrige.ok import OrdinaryKriging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LagrangianInterpolator:
    """
    A class to interpolate Lagrangian trajectory data to hourly positions
    using various interpolation methods including kriging. Currently only works
    with latitude, longitude, and time.
    """

    def __init__(self, data=None, lat_col='latitude', lon_col='longitude', time_col='time'):
        """
        Initialize the interpolator with optional data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing the Lagrangian data with lat, lon, and time columns
        lat_col : str
            Name of the latitude column
        lon_col : str
            Name of the longitude column
        time_col : str
            Name of the time column (should be convertible to datetime)
        """
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.time_col = time_col
        
        if data is not None:
            self.load_data(data)
    
    def load_data(self, data):
        """
        Load data into the interpolator and prepare it for processing.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing the Lagrangian data
        """
        self.data = data.copy()
        
        # Ensure time is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(self.data[self.time_col]):
            self.data[self.time_col] = pd.to_datetime(self.data[self.time_col])
        
        # Sort by time
        self.data = self.data.sort_values(by=self.time_col)
        
        # Convert times to numeric for interpolation
        self.data['time_numeric'] = (self.data[self.time_col] - self.data[self.time_col].min()).dt.total_seconds()
        
        print(f"Loaded {len(self.data)} data points spanning from "
              f"{self.data[self.time_col].min()} to {self.data[self.time_col].max()}")
    
    def interpolate_linear(self, target_times=None, hourly=True):
        """
        Perform linear interpolation to get positions at regular time intervals.
        
        Parameters:
        -----------
        target_times : list or pandas.Series
            List of target times for interpolation. If None, hourly times will be generated.
        hourly : bool
            If True and target_times is None, generate hourly times for interpolation
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with interpolated positions at target times
        """
        if target_times is None:
            # Generate hourly times if not provided
            start_time = self.data[self.time_col].min()
            end_time = self.data[self.time_col].max()
            
            if hourly:
                # Round to the nearest hour
                start_time = start_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                end_time = end_time.replace(minute=0, second=0, microsecond=0)
                
                # Generate hourly times
                target_times = pd.date_range(start=start_time, end=end_time, freq='H')
            else:
                # Use original times (for testing)
                target_times = self.data[self.time_col]
        
        # Convert target times to numeric for interpolation
        target_times_numeric = [(t - self.data[self.time_col].min()).total_seconds() for t in target_times]
        
        # Create interpolation functions for latitude and longitude
        f_lat = interp1d(self.data['time_numeric'], self.data[self.lat_col], 
                        bounds_error=False, fill_value="extrapolate")
        f_lon = interp1d(self.data['time_numeric'], self.data[self.lon_col], 
                        bounds_error=False, fill_value="extrapolate")
        
        # Interpolate at target times
        lat_interp = f_lat(target_times_numeric)
        lon_interp = f_lon(target_times_numeric)
        
        # Create result dataframe
        result = pd.DataFrame({
            self.time_col: target_times,
            self.lat_col: lat_interp,
            self.lon_col: lon_interp,
            'method': 'linear'
        })
        
        return result
    
    def interpolate_kriging(self, target_times=None, hourly=True, variogram_model='linear'):
        """
        Use Ordinary Kriging to interpolate positions.
        Note: This is a simplified implementation that treats time as a spatial dimension.
        For more complex spatiotemporal kriging, specialized libraries are recommended.
        
        Parameters:
        -----------
        target_times : list or pandas.Series
            List of target times for interpolation. If None, hourly times will be generated.
        hourly : bool
            If True and target_times is None, generate hourly times for interpolation
        variogram_model : str
            Variogram model to use for kriging ('linear', 'power', 'gaussian', 'spherical', 'exponential')
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with interpolated positions at target times
        """
        if target_times is None:
            # Generate hourly times if not provided
            start_time = self.data[self.time_col].min()
            end_time = self.data[self.time_col].max()
            
            if hourly:
                # Round to the nearest hour
                start_time = start_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                end_time = end_time.replace(minute=0, second=0, microsecond=0)
                
                # Generate hourly times
                target_times = pd.date_range(start=start_time, end=end_time, freq='H')
            else:
                # Use original times (for testing)
                target_times = self.data[self.time_col]
        
        # Convert target times to numeric for interpolation
        target_times_numeric = np.array([(t - self.data[self.time_col].min()).total_seconds() 
                                        for t in target_times])
        
        # Normalize time values to be on a similar scale as lat/lon
        time_scale = self.data['time_numeric'].max() / 100
        time_values_norm = self.data['time_numeric'] / time_scale
        target_times_norm = target_times_numeric / time_scale
        
        # Create the kriging models
        # For latitude
        OK_lat = OrdinaryKriging(
            time_values_norm,
            np.zeros_like(time_values_norm),  # Placeholder for second spatial dimension
            self.data[self.lat_col],
            variogram_model=variogram_model,
            verbose=False,
            enable_plotting=False
        )
        
        # For longitude
        OK_lon = OrdinaryKriging(
            time_values_norm,
            np.zeros_like(time_values_norm),  # Placeholder for second spatial dimension
            self.data[self.lon_col],
            variogram_model=variogram_model,
            verbose=False,
            enable_plotting=False
        )
        
        # Perform kriging
        lat_interp, lat_var = OK_lat.execute('points', target_times_norm, np.zeros_like(target_times_norm))
        lon_interp, lon_var = OK_lon.execute('points', target_times_norm, np.zeros_like(target_times_norm))
        
        # Create result dataframe
        result = pd.DataFrame({
            self.time_col: target_times,
            self.lat_col: lat_interp,
            self.lon_col: lon_interp,
            'lat_variance': lat_var,
            'lon_variance': lon_var,
            'method': 'kriging'
        })
        
        return result
    
    def interpolate_idw(self, target_times=None, hourly=True, power=2):
        """
        Use Inverse Distance Weighting (IDW) to interpolate positions.
        
        Parameters:
        -----------
        target_times : list or pandas.Series
            List of target times for interpolation. If None, hourly times will be generated.
        hourly : bool
            If True and target_times is None, generate hourly times for interpolation
        power : float
            Power parameter for IDW (higher values give more weight to closer points)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with interpolated positions at target times
        """
        if target_times is None:
            # Generate hourly times if not provided
            start_time = self.data[self.time_col].min()
            end_time = self.data[self.time_col].max()
            
            if hourly:
                # Round to the nearest hour
                start_time = start_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                end_time = end_time.replace(minute=0, second=0, microsecond=0)
                
                # Generate hourly times
                target_times = pd.date_range(start=start_time, end=end_time, freq='H')
            else:
                # Use original times (for testing)
                target_times = self.data[self.time_col]
        
        # Convert target times to numeric for interpolation
        target_times_numeric = np.array([(t - self.data[self.time_col].min()).total_seconds() 
                                       for t in target_times])
        
        # IDW implementation
        lat_interp = []
        lon_interp = []
        
        for target_time in target_times_numeric:
            # Calculate temporal distances
            distances = np.abs(self.data['time_numeric'] - target_time)
            
            # Handle exact matches to avoid division by zero
            if np.any(distances == 0):
                exact_matches = distances == 0
                lat_interp.append(np.mean(self.data.loc[exact_matches, self.lat_col]))
                lon_interp.append(np.mean(self.data.loc[exact_matches, self.lon_col]))
                continue
            
            # Calculate weights
            weights = 1.0 / (distances ** power)
            sum_weights = np.sum(weights)
            
            # Calculate weighted averages
            lat_val = np.sum(weights * self.data[self.lat_col].values) / sum_weights
            lon_val = np.sum(weights * self.data[self.lon_col].values) / sum_weights
            
            lat_interp.append(lat_val)
            lon_interp.append(lon_val)
        
        # Create result dataframe
        result = pd.DataFrame({
            self.time_col: target_times,
            self.lat_col: lat_interp,
            self.lon_col: lon_interp,
            'method': 'idw'
        })
        
        return result
    
    def plot_trajectories(self, original=True, interpolated=None, methods=None, basemap=False):
        """
        Plot original and interpolated trajectories.
        
        Parameters:
        -----------
        original : bool
            Whether to plot the original data points
        interpolated : pandas.DataFrame or list of pandas.DataFrame
            The interpolated data to plot
        methods : list of str
            Names of methods to include in the plot (if None, plot all)
        basemap : bool
            Whether to add a basemap (requires contextily package)
        """
        plt.figure(figsize=(12, 8))
        
        # Plot original data if requested
        if original:
            plt.scatter(self.data[self.lon_col], self.data[self.lat_col], color='k', 
                    label='Original data', s=8, alpha=0.7)
        
        # Plot interpolated data
        if interpolated is not None:
            # Convert to list if single DataFrame
            if isinstance(interpolated, pd.DataFrame):
                interpolated = [interpolated]
            
            for interp_df in interpolated:
                # Filter by methods if specified
                if methods is not None:
                    for method in methods:
                        subset = interp_df[interp_df['method'] == method]
                        if not subset.empty:
                            plt.scatter(subset[self.lon_col], subset[self.lat_col], s=8, 
                                   label=f'{method.capitalize()} interpolation', alpha=0.7)
                else:
                    # If multiple methods in one DataFrame, plot each separately
                    for method in interp_df['method'].unique():
                        subset = interp_df[interp_df['method'] == method]
                        plt.scatter(subset[self.lon_col], subset[self.lat_col], s=8,
                               label=f'{method.capitalize()} interpolation', alpha=0.7)
        
        # Add basemap if requested
        if basemap:
            try:
                import contextily as ctx
                ctx.add_basemap(plt.gca(), crs="EPSG:4326", source=ctx.providers.OpenStreetMap.Mapnik)
            except ImportError:
                print("contextily package not found. Install it to use basemaps.")
        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Lagrangian Trajectory Interpolation')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_3d_trajectory(self, original=True, interpolated=None, methods=None):
        """
        Create a 3D plot with time as the z-axis.
        
        Parameters:
        -----------
        original : bool
            Whether to plot the original data points
        interpolated : pandas.DataFrame or list of pandas.DataFrame
            The interpolated data to plot
        methods : list of str
            Names of methods to include in the plot (if None, plot all)
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot original data if requested
        if original:
            times_orig = (self.data[self.time_col] - self.data[self.time_col].min()).dt.total_seconds() / 3600  # hours
            ax.scatter(self.data[self.lon_col], self.data[self.lat_col], times_orig, 
                      c='k', label='Original data', s=50, alpha=0.7)
            ax.plot(self.data[self.lon_col], self.data[self.lat_col], times_orig, 'k-', alpha=0.5)
        
        # Plot interpolated data
        if interpolated is not None:
            # Convert to list if single DataFrame
            if isinstance(interpolated, pd.DataFrame):
                interpolated = [interpolated]
            
            colors = ['red', 'blue', 'green', 'purple', 'orange']
            color_idx = 0
            
            for interp_df in interpolated:
                # Convert times to hours for better visualization
                interp_df['time_hours'] = (interp_df[self.time_col] - self.data[self.time_col].min()).dt.total_seconds() / 3600
                
                # Filter by methods if specified
                if methods is not None:
                    for method in methods:
                        subset = interp_df[interp_df['method'] == method]
                        if not subset.empty:
                            ax.scatter(subset[self.lon_col], subset[self.lat_col], subset['time_hours'],
                                      c=colors[color_idx % len(colors)], label=f'{method.capitalize()} interpolation', s=30, alpha=0.7)
                            ax.plot(subset[self.lon_col], subset[self.lat_col], subset['time_hours'],
                                  c=colors[color_idx % len(colors)], alpha=0.5)
                            color_idx += 1
                else:
                    # If multiple methods in one DataFrame, plot each separately
                    for method in interp_df['method'].unique():
                        subset = interp_df[interp_df['method'] == method]
                        ax.scatter(subset[self.lon_col], subset[self.lat_col], subset['time_hours'],
                                  c=colors[color_idx % len(colors)], label=f'{method.capitalize()} interpolation', s=30, alpha=0.7)
                        ax.scatter(subset[self.lon_col], subset[self.lat_col], subset['time_hours'],
                              c=colors[color_idx % len(colors)], alpha=0.5)
                        color_idx += 1
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Time (hours)')
        ax.set_title('3D Lagrangian Trajectory (Time as Z-axis)')
        plt.legend()
        
        return fig

## Example usage
#if __name__ == "__main__":
#    # Create some example data
#    # In a real-world scenario, you would load this from a file or database
#    np.random.seed(42)
#    
#    # Generate random times (2 days, 4-hour intervals)
#    n_points = 12
#    start_time = datetime(2025, 5, 1, 0, 0, 0)
#    times = [start_time + timedelta(hours=4*i) for i in range(n_points)]
#    
#    # Generate random lat/lon that form a somewhat realistic trajectory
#    lons = np.cumsum(np.random.normal(0, 0.1, n_points)) - 122.0
#    lats = np.cumsum(np.random.normal(0, 0.08, n_points)) + 37.0
#    
#    # Create a dataframe
#    df = pd.DataFrame({
#        'time': times,
#        'latitude': lats,
#        'longitude': lons
#    })
#    
#    print("Example data:")
#    print(df)
#    
#    # Initialize the interpolator with the data
#    interpolator = LagrangianInterpolator(df)
#    
#    # Perform linear interpolation to hourly positions
#    linear_interp = interpolator.interpolate_linear()
#    
#    # Perform kriging interpolation
#    try:
#        kriging_interp = interpolator.interpolate_kriging(variogram_model='linear')
#        print("\nKriging interpolation successful")
#    except Exception as e:
#        print(f"\nKriging failed: {e}")
#        kriging_interp = None
#    
#    # Perform IDW interpolation
#    idw_interp = interpolator.interpolate_idw(power=2)
#    
#    # Combine all results
#    all_interp = pd.concat([linear_interp, 
#                          kriging_interp if kriging_interp is not None else pd.DataFrame(), 
#                          idw_interp])
#    
#    print("\nInterpolated results sample:")
#    print(all_interp.head())
#    
#    # Plot the results
#    interpolator.plot_trajectories(interpolated=all_interp)
#    plt.savefig('trajectory_comparison.png')
#    
#    # 3D plot
#    interpolator.plot_3d_trajectory(interpolated=all_interp)
#    plt.savefig('trajectory_3d.png')
#    
#    plt.show()