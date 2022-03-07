import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

class outlierPlotScatter:
    ''' This class will remove outliers and create the scatter plot
    '''
    
    def __init__(self, path):
        self.df = pd.read_csv(path)
    
    def removeOutliers(self) -> pd.DataFrame:
        """
        Removes outliers from a dataframe.
        """
        # Remove outliers
        self.df['lat_zscore'] = stats.zscore(self.df['latitude'])
        self.df['lon_zscore'] = stats.zscore(self.df['longitude'])
        return self.df[(self.df['lon_zscore'].abs()<3) & 
                       (self.df['lat_zscore'].abs()<3)].drop(['lat_zscore', 
                                                              'lon_zscore'], 
                                                             axis=1)

    def plotScatter(self, df: pd.DataFrame)-> None:
        plt.figure(figsize = (15,8))
        ax = sns.scatterplot(x = df['latitude'], y = df['longitude'], 
                            hue=df['occupancy'])
        ax.locator_params(axis = 'x', nbins=4)
        ax.locator_params(axis = 'y', nbins=3)
        plt.savefig('images/lat_lon_scatter.png')


class createTrainingData:
    ''' This class will create the training data to predict the next location
        of the cab.
    '''
    
    def __init__(self, df: pd.DataFrame, x_variable:str, no_of_steps:int,
                 training_data_size:int):
        self.df = df
        self.x_variable = x_variable
        self.no_of_steps = no_of_steps
        self.training_data_size = training_data_size
        
    def split_sequence(self):
        ''' This function will split the sequence into X and y datasets with a
            lag of n timesteps
        '''
        sequence = self.df[self.x_variable].values
        sequence = sequence[:self.training_data_size]
        X, y = [], []
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence)-1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
    
    def createTrainTestData(self):
        ''' This function will create the test train data. Since it is a time
            series prediction, the test data will be the last 20% of the array.
        '''
        X, y = self.split_sequence()
        
        train_size = round(len(X)*.8)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test   = X[train_size:], y[train_size:]
        
        X_train, y_train = X_train.astype(float), y_train.astype(float)
        X_test, y_test = X_test.astype(float), y_test.astype(float)
        
        return X_train ,X_test, y_train, y_test
    

class createModels:
    ''' This will define the deep learning model which is to be tested on the
        dataset.
    '''
    
    def __init__(self, X_train: np.array, X_test: np.array, 
                 y_train: np.array, y_test: np.array):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def createLSTMFeatureSpace(self):
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        n_features = 1
        lstm_X_train = self.X_train.reshape((self.X_train.shape[0], 
                                             self.X_train.shape[1], 
                                             n_features))
        lstm_X_test = self.X_test.reshape((self.X_test.shape[0], 
                                           self.X_test.shape[1], 
                                           n_features)) 
        return lstm_X_train, lstm_X_test

    def createDLModel(self):
        ''' This function will create the model.
        '''
        lstm_X_train, lstm_X_test = self.createLSTMFeatureSpace()
        model = Sequential()
        model.add(LSTM(50, input_shape=(lstm_X_train.shape[1], 
                                        lstm_X_train.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        return model
    
        
    def createGBModel(self):
        ''' This function will create the gradient boosting model.
        '''
        model = GradientBoostingRegressor(n_estimators=100, 
                                          learning_rate=0.1, 
                                          max_depth=1, 
                                          random_state=0, 
                                          loss='squared_error')
        return model

    def createXGBModel(self):
        ''' This function will create the XGBoost model.
        '''
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, 
                                max_depth=1, random_state=0)
        return model
    
    def trainXGBModel(self):
        ''' This function will train the XGBoost model.
        '''
        model = self.createXGBModel()
        model.fit(self.X_train, self.y_train)
        return model
    
    def trainDLModel(self):
        ''' This function will train the deep learning model.
        '''
        lstm_X_train, lstm_X_test = self.createLSTMFeatureSpace()
        model = self.createDLModel()
        history = model.fit(lstm_X_train, self.y_train, epochs=10, 
                            verbose=1, validation_split=0.1)
        return history, model, lstm_X_test
    
    def trainGBRModel(self):
        ''' This function will train the gradient boosting model.
        '''
        model = self.createGBModel()
        model.fit(self.X_train, self.y_train)
        return model
    

class evalHelper:
  ''' This will help in creating eval metrics and plots
  '''
  def __init__(self, y_test, DLmodel_pred, GBR_model_pred, XGB_model_pred):
    self.y_test = y_test
    self.DLmodel_pred = DLmodel_pred
    self.GBR_model_pred = GBR_model_pred
    self.XGB_model_pred = XGB_model_pred
  
  @staticmethod
  def createLossPlot(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.savefig('../images/LossPlot.png')
    
  def evaluationMetrics(self) -> dict:
    ''' This function will evaluate the models and return the mean absolute
        error.
    '''
    eval_dict ={}
    # DL model
    DL_MAE = mean_absolute_error(y_test, self.DLmodel_pred)
    DL_MSE = mean_squared_error(y_test, self.DLmodel_pred)
    print('MAE and MSE for DL model: ', DL_MAE, DL_MSE)
    
    # GBR model
    GBR_MAE = mean_absolute_error(y_test, self.GBR_model_pred)
    GBR_MSE = mean_squared_error(y_test, self.GBR_model_pred)
    print('MAE and MSE for GBR model: ', GBR_MAE, GBR_MSE)
    
    # XGB model
    XGB_MAE = mean_absolute_error(y_test, self.XGB_model_pred)
    XGB_MSE = mean_squared_error(y_test, self.XGB_model_pred)
    print('MAE and MSE for XGB model: ', XGB_MAE, XGB_MSE)
    
    eval_dict['DL_MAE'], eval_dict['DL_MSE'] = DL_MAE, DL_MSE
    eval_dict['GBR_MAE'], eval_dict['GBR_MSE'] = GBR_MAE, GBR_MSE
    eval_dict['XGB_MAE'], eval_dict['XGB_MSE'] = XGB_MAE, XGB_MSE
    
    return eval_dict

  def inverseTransform(self, minmaxscaler):
    ''' This function will inverse transform the data.
    '''
    Y_test_final = mms.inverse_transform(np.concatenate(
        [self.Y_test[0].reshape(-1,1), self.Y_test[1].reshape(-1,1)], axis=1))
    
    DL_model_final = mms.inverse_transform(np.concatenate(
        [self.DLmodel_pred[0].reshape(-1,1), self.DLmodel_pred[1].reshape(-1,1)], 
        axis=1))
    
    GBR_model_final = mms.inverse_transform(np.concatenate(
        [self.GBR_model_pred[0].reshape(-1,1), self.GBR_model_pred[1].reshape(-1,1)], 
        axis=1))
    
    XGB_model_final = mms.inverse_transform(np.concatenate(
        [self.XGB_model_pred[0].reshape(-1,1), self.XGB_model_pred[1].reshape(-1,1)], 
        axis=1))
    return Y_test_final, DL_model_final, GBR_model_final, XGB_model_final 
    
  @staticmethod
  def createScatterPlotResults(Y_test_final: np.array, 
                               DL_model_final: np.array,
                              GBR_model_final: np.array, 
                              XGB_model_final: np.array):
      plt.figure(figsize=(20, 20))
      plt.subplot(421)
      plt.scatter(Y_test_final[:,0], Y_test_final[:,-1])
      plt.title('True trajectory')
      plt.ylabel('Latitude')
      plt.xlabel('Longitude')
      
      plt.subplot(422)
      plt.scatter(DL_model_final[:,0], DL_model_final[:,-1])
      plt.title('LSTM Predicted trajectory')
      plt.ylabel('Latitude')
      plt.xlabel('Longitude')
      
      plt.subplot(423)
      plt.scatter(GBR_model_final[:,0], GBR_model_final[:,-1])
      plt.title('GBM Predicted trajectory')
      plt.ylabel('Latitude')
      plt.xlabel('Longitude')
      
      plt.subplot(424)
      plt.scatter(XGB_model_final[:,0], XGB_model_final[:,-1])
      plt.title('XGB Predicted trajectory')
      plt.ylabel('Latitude')
      plt.xlabel('Longitude')
      
      plt.savefig('../images/ScatterPlotResults.png')
      plt.tight_layout()

Y_test = []
DLmodel_pred = []
GBR_model_pred = []
XGB_model_pred = []
eval_results = []
hist_dl = []
outlier = outlierPlotScatter('../data/data.csv')
df_out = outlier.removeOutliers()
mms = MinMaxScaler()
df_out[['normalized_lat', 'normalized_lon']] = mms.fit_transform(
    df_out[['latitude', 'longitude']])

for item in ['normalized_lat', 'normalized_lon']:
    
    # Creating the training and test data
    createTD = createTrainingData(df_out, item, 3, 250000)
    X_train, X_test, y_train, y_test = createTD.createTrainTestData()
    
    # Instantiating the models and training them
    createModel = createModels(X_train, X_test, y_train, y_test)
    hist, DLmodel, lstm_X_test = createModel.trainDLModel()
    GBRModel = createModel.trainGBRModel()
    XGB = createModel.trainXGBModel()
    
    Y_test.append(y_test)
    dl_pred  = DLmodel.predict(lstm_X_test)
    gbr_pred = GBRModel.predict(X_test)
    xgb_pred = XGB.predict(X_test)

    DLmodel_pred.append(dl_pred)
    GBR_model_pred.append(gbr_pred)
    XGB_model_pred.append(xgb_pred)
    eval_cls = evalHelper(Y_test, dl_pred, gbr_pred, xgb_pred)
    eval_results.append(eval_cls.evaluationMetrics())
    hist_dl.append(hist)

y_true,dl_pred,gbr_pred,xgb_pred = eval_cls.inverseTransform(mms, 
                                                             Y_test, 
                                                             DLmodel_pred,
                                                             GBR_model_pred, 
                                                             XGB_model_pred)
eval_cls.createScatterPlotResults(y_true, dl_pred, gbr_pred, xgb_pred)
eval_cls.createLossPlot(hist_dl[0])
eval_cls.createLossPlot(hist_dl[1])