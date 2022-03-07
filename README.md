# CabSpottingAnalysis

The code is organized as follows:

1.  `_data_prep.py` - This will join all the text files and create and save a dataframe with all the rows in it.
2.  `_EDA.py` - This python file will do the robustness check on the data whether there are surprises in the data or not. It also creates and saves EDA plots.
3.  `_CO2_reduction.py` - Here the analysis for the reduction in the CO_2 emissions is done
4.  `_train_models_lat_long_predictor.py` - This python file creates the models to predict the next latitude and longitude for the cab
5.  `_occupancy_model.py` - This python file will create a model for predicting the occupancy of the taxi cab at a particular location and point in time.
6.  `_taxi_cabs_of_interest.py` - This python file will do the analysis to determine the high performers and low performers from the point of view of the taxi cab companies.

`requirements.txt` - The libraries which will be needed to run the above files or create a conda environment to run the above code

`sample_data` - some data files for test running the above code. The entire dataset can be downloaded from Kaggle.
