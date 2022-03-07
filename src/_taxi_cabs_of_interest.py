import pandas as pd
import numpy as np

class prepareTaxiData:
    ''' This will help in aggregating the data for the cabs to generate 
        insights into which cabs are important.
    '''
    
    def __init__(self, path):
        self.df = pd.read_csv(path)
        
    def createAggregatedData(self):
        ''' This will create the aggregated data for the cabs in terms of
            distance traveled as well as time on road.
        '''
        self.df['time'] = pd.to_datetime(self.df['time'])
        
        # getting the time on road for each data point entry
        self.df['time_on_road'] = self.df.sort_values(['cab_name','time']).\
            groupby('cab_name')['time'].diff()
        
        # Filling null values so as to aggregate and sum the time on road
        u = self.df.select_dtypes(include=['timedelta64[ns]'])
        self.df[u.columns] = u.fillna('0 days 00:00:00')
        self.df['time_on_road'] = self.df['time_on_road'].dt.total_seconds().\
            astype('int')
        
        # getting the distance traveled for each data point entry as well as 
        # total time on the road
        df_cab_sum = self.df.groupby(['cab_name', 'occupancy'])\
            .agg({'distance':'sum', 'time_on_road':'sum'}).reset_index()
        return df_cab_sum
    
    def getFinalDf(self):
        df_temp = self.createAggregatedData()
        self.df['date'] = self.df['time'].dt.date
        df_temp2 = self.df.groupby('cab_name').agg({'date':'nunique'}).reset_index()
        df_temp2.rename(columns={'date':'no_of_active_days'}, inplace=True)
        
        df_final = pd.merge(df_temp, df_temp2, on='cab_name')
        df_final['active_minutes_per_day'] = df_final['time_on_road']/\
            (df_final['no_of_active_days']*60)
        df_final['distance_per_day'] = df_final['distance']/\
            df_final['no_of_active_days']
        
        return df_final[['cab_name', 'distance_per_day', 
                         'active_minutes_per_day', 'occupancy']]
        
        
class importantCabs:
    ''' This will figure out the cabs which might be of interest to the
        taxi cab company. The high performers and the low performers.
    '''
    
    def __init__(self, df):
        self.df_occupied = df[df['occupancy']==1].drop(['occupancy'], axis=1)
        self.df_vaccant  = df[df['occupancy']==0].drop(['occupancy'], axis=1)

    def returnQuartiles(self, df):
        ''' This will return the quartile 1 and 3 for the dataframe passed as 
            argument.
        '''
        return [*df.quantile([.25, .75])]
    
    def getCabsOfInterest(self):
        ''' This will return the cabs which are of interest to us
        '''
        
        # getting the quartiles for the occupied and vaccant cabs
        Q1_dist_occ, Q3_dist_occ = self.returnQuartiles(
            self.df_occupied['distance_per_day'])
        Q1_time_occ, Q3_time_occ = self.returnQuartiles(
            self.df_occupied['active_minutes_per_day'])
        Q1_dist_vacc, Q3_dist_vacc = self.returnQuartiles(
            self.df_vaccant['distance_per_day'])
        Q1_time_vacc, Q3_time_vacc = self.returnQuartiles(
            self.df_vaccant['active_minutes_per_day'])
        
        fast_and_furious = self.df_occupied[
            (self.df_occupied['distance_per_day'] > Q3_dist_occ) &\
            (self.df_occupied['active_minutes_per_day'] < Q1_time_occ)]
        quick_doers = self.df_vaccant[
            (self.df_vaccant['distance_per_day'] < Q1_dist_vacc) &\
            (self.df_vaccant['active_minutes_per_day'] < Q1_time_vacc)]
        
        almost_lost_causes = self.df_vaccant[
            (self.df_vaccant['distance_per_day'] > Q3_dist_vacc) &\
            (self.df_vaccant['active_minutes_per_day'] > Q3_time_vacc)] 
        easy_on_the_pedal = self.df_occupied[
            (self.df_occupied['distance_per_day'] < Q1_dist_occ) &\
            (self.df_occupied['active_minutes_per_day'] > Q3_time_occ)]

        high_performers = np.concatenate([fast_and_furious['cab_name'].values, 
                                          quick_doers['cab_name'].values])
        low_performers  = np.concatenate([easy_on_the_pedal['cab_name'].values, 
                                          almost_lost_causes['cab_name'].values])
        high_performers = list(np.unique(high_performers))
        low_performers = list(np.unique(low_performers))
        return high_performers, low_performers 
        
    def getPerformers(self):    
        ''' There might be some high performers who are also present in low 
            performers. This might be becuase they have low average vacant 
            distance per day and low average active minutes per day vacant.
        '''
        hp_temp, lp_temp = self.getCabsOfInterest()
        lp_final = [cab for cab in lp_temp if cab not in hp_temp]
        print('The number of high performers are: ', len(hp_temp))
        print('The number of low performers are: ', len(lp_final))
        
        return hp_temp, lp_final
        
        
if __name__ == '__main__':
    taxidata = prepareTaxiData('data/combined_data.csv')
    df = taxidata.getFinalDf()
    
    cabsOfInterest = importantCabs(df)
    high_performers, low_performers = cabsOfInterest.getPerformers()


