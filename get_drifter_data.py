# Function taken from the original 'Drifter.py' script to download SVP drifter data from the erdap database

from urllib.parse import quote
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from datetime import datetime
import os
from glob import glob


def getdrift_csv() -> pd.DataFrame:
    """
        Function to get drifter data from url - return ids latitudes,longitudes,and times.
        input_time can either contain two values: start_time & end_time or one value as an interval
        example: input_time=[datetime(2012,1,1,0,0,0,0),datetime(2012,2,1,0,0,0,0)]
        """
    
    def check_files_drifter(path2downloads, newfile):
        newfile = newfile.split(os.path.sep)[-1]
        downloaded = [ff.split(os.path.sep)[-1] for ff in glob(os.path.join(path2downloads, '*.csv'))]
        if newfile in downloaded:
            return True
        else:
            return False

    path2save = os.path.join(os.getcwd(), 'RAW_DATA', 'DRIFTERS')
    
    platform_code = '"5501712|5501713|5501714|5501715|5501716|5501717|5501718|5501719|5501720|5501721|6801910|7801732|2802102|5802100|5802099|6801911|3801705|5802097|6801912|2802103"'
    today = datetime.now()
    input_time = [datetime(2023,1,1,0,0,0,0), datetime(today.year, today.month, today.day, today.hour, 0, 0, 0)]
    mintime=(input_time[0].strftime('%Y-%m-%d'+'T'+'%H')+':00:00')  # change time format to match ERDAP
    maxtime=(input_time[1].strftime('%Y-%m-%d'+'T'+'%H')+':00:00')
    filename = path2save + '/drifters_to_'+input_time[1].strftime('%Y-%m-%d'+'T'+'%H')

    downloaded = check_files_drifter(path2save, filename)
    if downloaded is True:
        print('File %s already exists', filename)
        drifter_df = pd.read_csv(filename)
    else:
        # construct url to get data
        baseurl = 'https://osmc.noaa.gov/erddap/tabledap/OSMC_30day.csv?'
        rawquery = ('platform_code,platform_type,time,latitude,longitude,sst,slp,uo,vo&platform_code=~'
                    +platform_code
                    +'&time>='+str(mintime)+'&time<='+str(maxtime)
                    +'&orderBy("platform_code")')
    
        # percent encode url and print raw query, encoded query, and access url
        query = quote(rawquery, encoding='utf-8', safe='&()=')
        url = baseurl+query
        print('Downloading drifter data from:' +url)
    
        # access url as a csv table loaded into a pandas dataframe
        drifter_df = pd.read_csv(url, skiprows=1) # Have to rename because I stuffed the URL up I think (the variables are the same thoug
        drifter_df = drifter_df.rename(columns={'Unnamed: 0':'ID','Unnamed: 1':'platform_type',
                                                'UTC':'time','degrees_north':'latitude','degrees_east':'longitude',
                                                'Deg C':'sst','m s-1':'vn','m s-1.1':'ve'})
    
        drifter_df.to_csv(path2save+'/maxtime.csv')

    return drifter_df