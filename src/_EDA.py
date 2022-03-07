import pandas as pd
import matplotlib.pyplot as plt

class robustnessData: 
    ''' The checkLatAndLong, checkOccupancy and checkTime functions are
        perfromed on the original data provided. The rest are all derived from
        the data given. Thus if the original data pass the checks and balances
        listed below the rest should also pass.
    '''
    
    def __init__(self, path: str):
        self.df = pd.read_csv(path)
        
    def checkLatAndLong(self) -> None:
        ''' The latitude and longitude have geographical limits of 
            [-90, 90] and [-180, 180] respectively. This function will remove
            the rows with invalid latitude and longitude.
        '''
        self.df[['latitude', 'longitude']].apply(lambda x: 
            x.clip(lower=-90, upper=90))
        self.df[['latitude', 'longitude']].apply(lambda x: 
            x.clip(lower=-180, upper=180))
    
    def checkOccupancy(self) -> None:
        ''' This function will remove the rows with invalid occupancy.
        '''
        self.df = self.df[self.df['occupancy'].isin([0, 1])]
    
    def checkTime(self) -> None:
        ''' This function will remove the rows with invalid time.
        '''
        self.df = self.df[self.df['time'].str.contains('[0-9]{2}:[0-9]{2}')]
    
    def save(self, path: str) -> None:
        ''' This function will save the cleaned dataframe to a csv file.
        '''
        self.checkLatAndLong()
        self.checkOccupancy()
        self.checkTime()
        print('Completed Robustness Check...! Now saving...')
        self.df.to_csv(path, index=False)
        

class plotter:
    ''' This class will plot all the explanatory graphs part of the EDA.
    '''
    def __init__(self, path: str):
        self.path = path
        self.df = pd.read_csv(self.path)

    def plotDistanceChartPie(self) -> None:
        ''' This creates Pie chart showing the occupancy rate of the cabs
        '''
        print('Total Km travelled by cabs: ', 
              round(self.df['distance'].sum(),0))
        print('Plotting the distance pie chart...')
        occupancy_dict ={0:'Vacant', 1:'Occupied'}
        occupancy_rate = self.df.groupby('occupancy').sum()['distance']
        occupancy_labels = [occupancy_dict[i] for i in occupancy_rate.index]
        plt.pie(occupancy_rate, labels=occupancy_labels, 
                colors=['darksalmon', 'mediumseagreen'], autopct='%1.0f%%')
        my_circle=plt.Circle( (0,0), 0.75, color='white')
        p=plt.gcf()
        p.gca().add_artist(my_circle)
        plt.savefig('./images/occupancy_rate.png')
        
    def plotDistanceWeekdayChart(self) -> None:
        ''' This creates a bar chart showing the distance travelled by cabs 
            on each day of the week.
        '''
        print('Plotting the distance travelled by cabs on each day of the week')
        self.df_temp = self.df.groupby(['occupancy', 'weekday']).sum()\
            ['distance'].unstack(level=0)
        occupancy_dict ={0:'Vacant', 1:'Occupied'}
        days = {1:'Monday', 2:'Tuesday', 3:'Wednesday', 4:'Thursday', 
                5:'Friday', 6:'Saturday', 7:'Sunday'}

        ax = self.df_temp.plot(kind='bar', stacked=False, figsize=(10,6),
                        color = ['darksalmon', 'mediumseagreen'])
        ax.set_xticklabels([days[i] for i in self.df_temp.index], rotation=45)
        ax.set_ylabel('Distance travelled (km)')
        ax.set_xlabel('Weekday')
        
        ax.grid(False)
        ax.legend(occupancy_dict.values(), loc='upper center', 
                bbox_to_anchor=(0.5, 1.1))
        plt.savefig('./images/distance_travelled_by_occupancy.png')
            
    def plotDistanceHourChart(self) -> None:
        ''' This function creates plot of the proportion of occupancy in the
            sum of total distance covered in each hour over the whole month.
        '''
        print('Plotting the distance travelled by cabs in each hour of the day')
        self.df_temp = self.df.groupby(['occupancy', 'hour']).sum()\
            ['distance'].unstack(level=0)
        self.df_temp.columns = ['Vacant', 'Occupied']
        for index, row in self.df_temp.iterrows():
            row[0], row[1] = round(row[0]/(row[0] + row[1]), 2)*100,\
                round(row[1]/(row[0] + row[1]), 2)*100
        ax = self.df_temp.plot(kind='bar', stacked=True, figsize=(12,7), 
                    color = ['darksalmon', 'mediumseagreen'])
        ax.grid(False)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1))
        plt.savefig('./images/occupancy_by_hour.png')
        
    def savePlots(self) -> None:
        ''' This function will save all the plots created.
        '''
        self.plotDistanceChartPie()
        self.plotDistanceWeekdayChart()
        self.plotDistanceHourChart()
        print('COmpleted all plots! Exiting...')
        
if __name__ == '__main__':
    csv_path = 'data/combined_data.csv'
    print('Reading in the data...')
    robust = robustnessData(csv_path)
    robust.save(csv_path)
    plotter = plotter(csv_path)
    plotter.savePlots()
