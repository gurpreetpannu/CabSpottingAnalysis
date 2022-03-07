import pandas as pd
from math import pow
import matplotlib.pyplot as plt
import numpy as np

class CO2Reduction:
    ''' This class will calculate the possible CO2 reduction per km when
        changing the fleet to electric
    '''
    
    def __init__(self, path: str):
        self.df = pd.read_csv(path)
        
    def extrapolateCleanKm(self) -> None:
        ''' This function will extrapolate the km travelled by the cabs
            based on the occupancy of the cabs.
        '''
        total_vacant_km=self.df[self.df['occupancy']==0]['distance'].sum() 
        
        # Working under the assumption that in month 1 all the cabs are IC
        # Now 15% of the cabs switch to electric every month
        clean_km_per_month = []
        for i in range(12):
            clean_km_per_month.append(total_vacant_km*pow(0.85,i)*0.15)
        return clean_km_per_month, total_vacant_km
    
    def co2EmissionSaved(self, clean_km_per_month: list, 
                         renewable_energy=False) -> list:
        ''' This function will calculate the CO2 emission saved by the fleet
            when switching to electric
        '''
        emission_per_km = 404/1.60934
        co2_emission_saved = []
        for i in range(len(clean_km_per_month)):
            if renewable_energy:
                co2_emission_saved.append(round(clean_km_per_month[i]*
                                                emission_per_km, 1))
            else:
                co2_emission_saved.append(round(clean_km_per_month[i]*
                                        emission_per_km*0.64,2))
            co2_emission_saved
        return [round(i/1000000, 1) for i in co2_emission_saved]
    
    def plotCO2Saved(self, co2_saved_renewable: list,
                 co2_saved_grid: list) -> None:
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                  'Sep', 'Oct', 'Nov', 'Dec']
        additional = [i-j for i,j in zip(co2_saved_renewable, co2_saved_grid)]
        plt.stackplot(months, co2_saved_grid, additional,
                      colors=['coral', 'yellowgreen'], 
                      labels=['Grid', 'Renewable'] )
        plt.ylabel('CO2 Emission Saved (tonnes)')
        plt.title('Additional CO2 Emission Saved by Cabs Switching to Electric')
        plt.legend(loc='upper right')
        plt.savefig('../images/co2_saved.png')     
        
           
if __name__ == '__main__':
    path = '../sample_data/data.csv'
    co2_reduction = CO2Reduction(path)
    print('Reading in the data and calculating clean km per month')
    clean_km_per_month, total_vkm = co2_reduction.extrapolateCleanKm()

    print('total yearly co2_emission:', round((total_vkm*(404/1.60934))*12,1))
    
    print('Calculting CO2 emission saved when cabs charged on grid')
    co2_saved_grid = co2_reduction.co2EmissionSaved(clean_km_per_month)
    
    print('Calculating CO2 emission saved when cabs charged on renewable')
    co2_saved_renewable = co2_reduction.co2EmissionSaved(clean_km_per_month,
                                                         renewable_energy=True)
    
    print('Plotting the CO2 emission saved')
    co2_reduction.plotCO2Saved(co2_saved_renewable, co2_saved_grid)
    
    print('Total CO2 emission saved by switching to electric when charging cars on the grid:',round(sum(np.cumsum(co2_saved_grid)),1))
    print('Total CO2 emission saved by switching to electric when charging cars on the renewable:',round(sum(np.cumsum(co2_saved_renewable)),1))
