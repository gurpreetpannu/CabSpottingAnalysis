
from glob import glob
import pandas as pd
import numpy as np
from datetime import datetime
import math
from scipy.ndimage.interpolation import shift
from tqdm import tqdm

class data_preparation:
    
    def __init__(self, files_path):
        self.files_list = glob(files_path+'*.txt')
        self.df_combined = pd.DataFrame()
        
    def read_individual_file(self, file):
        df = pd.read_csv(file, sep = ' ', header=None)
        df.columns = ['latitude', 'longitude', 'occupancy', 'time']
        cab_name = file.split('/')[-1].split('.')[0]
        df['cab_name'] = cab_name
        df['time'] = df['time'].apply(lambda x: datetime.fromtimestamp(x))
        return df
    
    # function to calculate distance between two points using haversine formula
    def haversine(self, lat1: float, lon1: float, 
                  lat2: float, lon2: float) -> float:
        ''' This function takes in latitude and longitude of two points and 
            returns the distance between them in km. 
        '''
        R = 6371 # Radius of the earth in km
        dLat = math.radians(lat2-lat1)
        dLon = math.radians(lon2-lon1)
        a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(math.radians(lat1)) \
            * math.cos(math.radians(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        d = R * c
        return d
    
    def combine_files(self) -> pd.DataFrame:
        for i in tqdm(range(len(self.files_list))):
            df = self.read_individual_file(self.files_list[i])
            self.df_combined = self.df_combined.append(df)
        return self.df_combined
       
    def calculateDistance(self):
        ''' This function will calculate distance between two consectutive row
            entries in the pandas dataframe by using native numpy methods.
        ''' 
        lat_array  = np.array(self.df_combined['latitude'].values)
        long_array = np.array(self.df_combined['longitude'].values)
        lat_array_shifted  = shift(lat_array, 1, cval=0)
        long_array_shifted = shift(long_array, 1, cval=0)
        total_array = np.array([lat_array, long_array, lat_array_shifted, 
                        long_array_shifted], dtype=object)
        print(lat_array.shape, long_array.shape, lat_array_shifted.shape,long_array_shifted.shape)
        dist = []
        for i in tqdm(range(total_array.shape[1])):
            dist.append(self.haversine(total_array[0][i], total_array[1][i],
                                       total_array[2][i], total_array[3][i]))
        self.df_combined['distance'] = dist
    
    def makeDistanceAdjustments(self): 
        ''' This fucntion makes changes to the distance column because I have 
            treated the latitudes and logitudes as continous between individual
            cab data. Thus last coordinates of one cab data file is used to 
            calculate distance for the next cab data file.
            Thus the first data point of each individual cab file should be 0.
        '''
        self.df_combined['distance'] = self.df_combined['distance'].round(3)
        cab_names = [*self.df_combined['cab_name'].unique()]
        idx = []
        for i in tqdm(range(len(cab_names))):
            idx.append(self.df_combined[self.df_combined['cab_name'] 
                                        == cab_names[i]].index[0])
        self.df_combined.loc[idx, 'distance'] = 0
       
    def makeTimeFeatures(self):
        ''' This function will generate some time features which might help
            in further anlaysis.
        '''    
        self.df_combined['hour'] = self.df_combined['time'].apply(
            lambda x: x.hour)
        self.df_combined['weekday'] = self.df_combined['time'].apply(
            lambda x: x.isoweekday())
        self.df_combined['is_weekend'] = np.where(
            self.df_combined['weekday'] >= 5, 1, 0)
        self.df_combined['office_rush'] = np.where(
            self.df_combined['hour'].isin([8,9,17,18]), 1, 0)
        self.df_combined['dinner_time'] = np.where(
            self.df_combined['hour'].isin([19,20,21,22]), 1, 0)
        self.df_combined['late_night'] = np.where(
            self.df_combined['hour'].isin([23,0,1,2,3,4,5]), 1, 0)
    
    def createSaveData(self):
        self.combine_files()
        print('Combined files')
        self.calculateDistance()
        print('Calculated distance')
        self.makeDistanceAdjustments()
        print('Adjusted distance')
        self.makeTimeFeatures()
        print('Made time features')
        print('The size of the final csv is {}'.format(self.df_combined.shape))
        self.df_combined.to_csv('../sample_data/data.csv', index=False)
        print('Saved combined data')        

if __name__ == '__main__':
    files_path = '../sample_data/'
    data_prep = data_preparation(files_path)
    data_prep.createSaveData()


