import xarray as xr
import numpy as np
from datetime import datetime
import pandas as pd
from tqdm import tqdm

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
        #self.location_type = np.zeros(nb_traj, dtype='bool') # 0 Argos, 1 GPS
        #self.wmo = np.zeros(nb_traj, dtype='int32')
        #self.expno = np.zeros(nb_traj, dtype='int32')
        #self.deploy_date = np.zeros(nb_traj, dtype='datetime64[s]')
        self.deploy_lat = np.zeros(nb_traj, dtype='float32')
        self.deploy_lon = np.zeros(nb_traj, dtype='float32')
        #self.end_date = np.zeros(nb_traj, dtype='datetime64[s]')
        #self.end_lon = np.zeros(nb_traj, dtype='float32')
        #self.end_lat = np.zeros(nb_traj, dtype='float32')
        #self.drogue_lost_date = np.zeros(nb_traj, dtype='datetime64[s]')
        #self.type_death = np.zeros(nb_traj, dtype='int8')
        #self.type_buoy = np.chararray(nb_traj, itemsize=15)
        #self.deployment_ship = np.chararray(nb_traj, itemsize=15)
        #self.deployment_status = np.chararray(nb_traj, itemsize=15)
        #self.buoy_type_manufacturer = np.chararray(nb_traj, itemsize=15)
        #self.buoy_type_sensor_array = np.chararray(nb_traj, itemsize=15)
        #self.current_program = np.zeros(nb_traj, dtype='int32')
        #self.purchaser_funding = np.chararray(nb_traj, itemsize=15)
        #self.sensor_upgrade = np.chararray(nb_traj, itemsize=15)
        #self.transmissions = np.chararray(nb_traj, itemsize=15)
        #self.deploying_country = np.chararray(nb_traj, itemsize=15)
        #self.deployment_comments = np.chararray(nb_traj, itemsize=15)
        #self.manufacture_year = np.zeros(nb_traj, dtype='int16')
        #self.manufacture_month = np.zeros(nb_traj, dtype='int16')
        #self.manufacture_sensor_type = np.chararray(nb_traj, itemsize=5)
        #self.manufacture_voltage = np.zeros(nb_traj, dtype='int16')
        #self.float_diameter = np.zeros(nb_traj, dtype='float32')
        #self.subsfc_float_presence = np.zeros(nb_traj, dtype='bool')
        #self.drogue_type = np.chararray(nb_traj, itemsize=15)
        #self.drogue_length = np.zeros(nb_traj, dtype='float32')
        #self.drogue_ballast = np.zeros(nb_traj, dtype='float32')
        #self.drag_area_above_drogue = np.zeros(nb_traj, dtype='float32')
        #self.drag_area_drogue = np.zeros(nb_traj, dtype='float32')
        #self.drag_area_ratio = np.zeros(nb_traj, dtype='float32')
        #self.drag_center_depth = np.zeros(nb_traj, dtype='float32')
        #self.drogue_detect_sensor = np.chararray(nb_traj, itemsize=15)

        # values defined at every observation (timestep)
        self.longitude = np.zeros(nb_obs, dtype='float32')
        self.latitude = np.zeros(nb_obs, dtype='float32')
        self.time = np.zeros(nb_obs, dtype='datetime64[s]')
        self.ve = np.zeros(nb_obs, dtype='float32')
        self.vn = np.zeros(nb_obs, dtype='float32')
        #self.err_lat = np.zeros(nb_obs, dtype='float32')
        #self.err_lon = np.zeros(nb_obs, dtype='float32')
        #self.err_ve = np.zeros(nb_obs, dtype='float32')
        #self.err_vn = np.zeros(nb_obs, dtype='float32')
        #self.gap = np.zeros(nb_obs, dtype='float32')
        #self.drogue_status = np.zeros(nb_obs, dtype='bool') # 1 drogued, 0 undrogued

        # sst data set
        self.sst = np.zeros(nb_obs, dtype='float32')
        #self.sst1 = np.zeros(nb_obs, dtype='float32')
        #self.sst2 = np.zeros(nb_obs, dtype='float32')
        #self.err_sst = np.zeros(nb_obs, dtype='float32')
        #self.err_sst1 = np.zeros(nb_obs, dtype='float32')
        #self.err_sst2 = np.zeros(nb_obs, dtype='float32')
        #self.flg_sst = np.zeros(nb_obs, dtype='int8')
        #self.flg_sst1 = np.zeros(nb_obs, dtype='int8')
        #self.flg_sst2 = np.zeros(nb_obs, dtype='int8')

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
        #self.wmo[tid] = ds.WMO.data[0]
        #self.expno[tid] = ds.expno.data[0]
        #self.deploy_date[tid] = decode_date(ds.deploy_date.data[0])
        self.deploy_lon[tid] = ds.deploy_lon.data[0]
        self.deploy_lat[tid] = ds.deploy_lat.data[0]
        #self.end_date[tid] = decode_date(ds.end_date.data[0])
        #self.end_lon[tid] = ds.end_lon.data[0]
        #self.end_lat[tid] = ds.end_lat.data[0]
        #self.drogue_lost_date[tid] = decode_date(ds.drogue_lost_date.data[0])
        #self.type_death[tid] = ds.typedeath.data[0]
        #self.type_buoy[tid] = ds.typebuoy.data[0]

        # vectors
        self.longitude[oid:oid+size] = ds.longitude.data[0]
        self.latitude[oid:oid+size] = ds.latitude.data[0]
        self.time[oid:oid+size] = (ds.time.data[0])
        self.ve[oid:oid+size] = ds.ve.data[0]
        self.vn[oid:oid+size] = ds.vn.data[0]
        #self.err_lat[oid:oid+size] = ds.err_lat.data[0]
        #self.err_lon[oid:oid+size] = ds.err_lon.data[0]
        #self.err_ve[oid:oid+size] = ds.err_ve.data[0]
        #self.err_vn[oid:oid+size] = ds.err_vn.data[0]
        #self.gap[oid:oid+size] = ds.gap.data[0]
        self.sst[oid:oid+size] = (ds.sst.data[0])
        #self.sst1[oid:oid+size] = fill_values(ds.sst1.data[0])
        #self.sst2[oid:oid+size] = fill_values(ds.sst2.data[0])
        #self.err_sst[oid:oid+size] = fill_values(ds.err_sst.data[0])
        #self.err_sst1[oid:oid+size] = fill_values(ds.err_sst1.data[0])
        #self.err_sst2[oid:oid+size] = fill_values(ds.err_sst2.data[0])
        #self.flg_sst[oid:oid+size] = ds.flg_sst.data[0]
        #self.flg_sst1[oid:oid+size] = ds.flg_sst1.data[0]
        #self.flg_sst2[oid:oid+size] = ds.flg_sst2.data[0]
        #self.drogue_status[oid:oid+size] = drogue_presence(self.drogue_lost_date[tid], self.time[oid:oid+size])

    def to_xarray(self):
        ds = xr.Dataset(
            data_vars=dict(
                rowsize=(['traj'], self.rowsize, {'long_name': 'Number of observations per trajectory', 'sample_dimension': 'obs', 'units':'-'}),
                #location_type=(['traj'], self.location_type, {'long_name': 'Satellite-based location system', 'units':'-', 'comments':'0 (Argos), 1 (GPS)'}),
                #WMO=(['traj'], self.wmo, {'long_name': 'World Meteorological Organization buoy identification number', 'units':'-'}),
                #expno=(['traj'], self.expno, {'long_name': 'Experiment number', 'units':'-'}),
                #deploy_date=(['traj'], self.deploy_date, {'long_name': 'Deployment date and time'}),
                deploy_lon=(['traj'], self.deploy_lon, {'long_name': 'Deployment longitude', 'units':'degrees_east'}),
                deploy_lat=(['traj'], self.deploy_lat, {'long_name': 'Deployment latitude', 'units':'degrees_north'}),
                #end_date=(['traj'], self.end_date, {'long_name': 'End date and time'}),
                #end_lat=(['traj'], self.end_lat, {'long_name': 'End latitude', 'units':'degrees_north'}),
                #end_lon=(['traj'], self.end_lon, {'long_name': 'End longitude', 'units':'degrees_east'}),
                #drogue_lost_date=(['traj'], self.drogue_lost_date, {'long_name': 'Date and time of drogue loss'}),
                #type_death=(['traj'], self.type_death, {'long_name': 'Type of death', 'units':'-', 'comments': '0 (buoy still alive), 1 (buoy ran aground), 2 (picked up by vessel), 3 (stop transmitting), 4 (sporadic transmissions), 5 (bad batteries), 6 (inactive status)'}),
                #type_buoy=(['traj'], self.type_buoy, {'long_name': 'Buoy type (see https://www.aoml.noaa.gov/phod/dac/dirall.html)', 'units':'-'}),
                #DeploymentShip=(['traj'], self.deployment_ship, {'long_name': 'Name of deployment ship', 'units':'-'}),
                #DeploymentStatus=(['traj'], self.deployment_status, {'long_name': 'Deployment status', 'units':'-'}),
                #BuoyTypeManufacturer=(['traj'], self.buoy_type_manufacturer, {'long_name': 'Buoy type manufacturer', 'units':'-'}),
                #BuoyTypeSensorArray=(['traj'], self.buoy_type_sensor_array, {'long_name': 'Buoy type sensor array', 'units':'-'}),
                #CurrentProgram=(['traj'], self.current_program, {'long_name': 'Current Program', 'units':'-', '_FillValue': '-1'}),
                #PurchaserFunding=(['traj'], self.purchaser_funding, {'long_name': 'Purchaser funding', 'units':'-'}),
                #SensorUpgrade=(['traj'], self.sensor_upgrade, {'long_name': 'Sensor upgrade', 'units':'-'}),
                #Transmissions=(['traj'], self.transmissions, {'long_name': 'Transmissions', 'units':'-'}),
                #DeployingCountry=(['traj'], self.deploying_country, {'long_name': 'Deploying country', 'units':'-'}),
                #DeploymentComments=(['traj'], self.deployment_comments, {'long_name': 'Deployment comments', 'units':'-'}),
                #ManufactureYear=(['traj'], self.manufacture_year, {'long_name': 'Manufacture year', 'units':'-', '_FillValue': '-1'}),
                #ManufactureMonth=(['traj'], self.manufacture_month, {'long_name': 'Manufacture month', 'units':'-', '_FillValue': '-1'}),
                #ManufactureSensorType=(['traj'], self.manufacture_sensor_type, {'long_name': 'Manufacture Sensor Type', 'units':'-'}),
                #ManufactureVoltage=(['traj'], self.manufacture_voltage, {'long_name': 'Manufacture voltage', 'units':'-', '_FillValue': '-1'}),
                #FloatDiameter=(['traj'], self.float_diameter, {'long_name': 'Diameter of surface floater', 'units':'cm'}),
                #SubsfcFloatPresence=(['traj'], self.subsfc_float_presence, {'long_name': 'Subsurface Float Presence', 'units':'-'}),
                #DrogueType=(['traj'], self.type_buoy, {'drogue_type': 'Drogue Type', 'units':'-'}),
                #DrogueLength=(['traj'], self.drogue_length, {'long_name': 'Length of drogue.', 'units':'m'}),
                #DrogueBallast=(['traj'], self.drogue_ballast, {'long_name': "Weight of the drogue's ballast.", 'units':'kg'}),
                #DragAreaAboveDrogue=(['traj'], self.drag_area_above_drogue, {'long_name': 'Drag area above drogue.', 'units':'m^2'}),
                #DragAreaOfDrogue=(['traj'], self.drag_area_drogue, {'long_name': 'Drag area drogue.', 'units':'m^2'}),
                #DragAreaRatio=(['traj'], self.drag_area_ratio, {'long_name': 'Drag area ratio', 'units':'m'}),
                #DrogueCenterDepth=(['traj'], self.drag_center_depth, {'long_name': 'Average depth of the drogue.', 'units':'m'}),
                #DrogueDetectSensor=(['traj'], self.drogue_detect_sensor, {'long_name': 'Drogue detection sensor', 'units':'-'}),

                # position and velocity
                ve=(['obs'], self.ve, {'long_name': 'Eastward velocity', 'units':'m/s'}),
                vn=(['obs'], self.vn, {'long_name': 'Northward velocity', 'units':'m/s'}),
                #gap=(['obs'], self.gap, {'long_name': 'Time interval between previous and next location', 'units':'s'}),
                #err_lat=(['obs'], self.err_lat, {'long_name': '95% confidence interval in latitude', 'units':'degrees_north'}),
                #err_lon=(['obs'], self.err_lon, {'long_name': '95% confidence interval in longitude', 'units':'degrees_east'}),
                #err_ve=(['obs'], self.err_ve, {'long_name': '95% confidence interval in eastward velocity', 'units':'m/s'}),
                #err_vn=(['obs'], self.err_vn, {'long_name': '95% confidence interval in northward velocity', 'units':'m/s'}),
                #drogue_status=(['obs'], self.drogue_status, {'long_name': 'Status indicating the presence of the drogue', 'units':'-', 'flag_values':'1,0', 'flag_meanings': 'drogued, undrogued'}),

                # sst
                sst=(['obs'], self.sst, {'long_name': 'Fitted sea water temperature', 'units':'Kelvin', 'comments': 'Estimated near-surface sea water temperature from drifting buoy measurements. It is the sum of the fitted near-surface non-diurnal sea water temperature and fitted diurnal sea water temperature anomaly. Discrepancies may occur because of rounding.'}),
                #sst1=(['obs'], self.sst1, {'long_name': 'Fitted non-diurnal sea water temperature', 'units':'Kelvin', 'comments': 'Estimated near-surface non-diurnal sea water temperature from drifting buoy measurements'}),
                #sst2=(['obs'], self.sst2, {'long_name': 'Fitted diurnal sea water temperature anomaly', 'units':'Kelvin', 'comments': 'Estimated near-surface diurnal sea water temperature anomaly from drifting buoy measurements'}),
                #err_sst=(['obs'], self.err_sst, {'long_name': 'Standard uncertainty of fitted sea water temperature', 'units':'Kelvin', 'comments': 'Estimated one standard error of near-surface sea water temperature estimate from drifting buoy measurements'}),
                #err_sst1=(['obs'], self.err_sst1, {'long_name': 'Standard uncertainty of fitted non-diurnal sea water temperature', 'units':'Kelvin', 'comments': 'Estimated one standard error of near-surface non-diurnal sea water temperature estimate from drifting buoy measurements'}),
                #err_sst2=(['obs'], self.err_sst2, {'long_name': 'Standard uncertainty of fitted diurnal sea water temperature anomaly', 'units':'Kelvin', 'comments': 'Estimated one standard error of near-surface diurnal sea water temperature anomaly estimate from drifting buoy measurements'}),
                #flg_sst=(['obs'], self.flg_sst, {'long_name': 'Fitted sea water temperature quality flag', 'units':'-', 'flag_values':'0, 1, 2, 3, 4, 5', 'flag_meanings': 'no-estimate, no-uncertainty-estimate, estimate-not-in-range-uncertainty-not-in-range, estimate-not-in-range-uncertainty-in-range estimate-in-range-uncertainty-not-in-range, estimate-in-range-uncertainty-in-range'}),
                #flg_sst1=(['obs'], self.flg_sst1, {'long_name': 'Fitted non-diurnal sea water temperature quality flag', 'units':'-', 'flag_values':'0, 1, 2, 3, 4, 5', 'flag_meanings': 'no-estimate, no-uncertainty-estimate, estimate-not-in-range-uncertainty-not-in-range, estimate-not-in-range-uncertainty-in-range estimate-in-range-uncertainty-not-in-range, estimate-in-range-uncertainty-in-range'}),
                #flg_sst2=(['obs'], self.flg_sst2, {'long_name': 'Fitted diurnal sea water temperature anomaly quality flag', 'units':'-', 'flag_values':'0, 1, 2, 3, 4, 5', 'flag_meanings': 'no-estimate, no-uncertainty-estimate, estimate-not-in-range-uncertainty-not-in-range, estimate-not-in-range-uncertainty-in-range estimate-in-range-uncertainty-not-in-range, estimate-in-range-uncertainty-in-range'}),
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