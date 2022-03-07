import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import xgboost as xgb


class prepareData:
    ''' This will prepare the train and test dataset
    '''
    
    def __init__(self, path, train_size=0.1):
        self.df = pd.read_csv(path)
        self.train_size = train_size
        
    def createFeatures(self):
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.df['minute'] = self.df['time'].dt.minute
        self.df['date'] = self.df['time'].dt.date
        self.df['is_holiday'] = np.where(self.df['date'] == 
                                         pd.to_datetime('2008-05-26').date(), 
                                         1, 0)
        
    def normalise(self):
        mms = MinMaxScaler()
        self.df[['normalised_lat', 'normalised_long']] = mms.fit_transform(
            self.df[['latitude', 'longitude']])
    
    def createTestTrainSplit(self):
        self.createFeatures()
        self.normalise()
        y = self.df['occupancy'].values
        X = self.df.drop(['occupancy', 'time', 'date', 'cab_name', 
                          'latitude', 'longitude'], axis=1).values
        X_subset = X[:self.train_size*len(X)]
        y_subset = y[:self.train_size*len(X)]
        X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, 
                                                            test_size=0.2, 
                                                            random_state=42)
        return X_train, X_test, y_train, y_test
        
class model:
    ''' This will create the models to be tested for predicting occupancy in 
        the cab.
    '''
    
    def __init__(self, X_train: np.array, X_test: np.array, 
                 y_train: np.array, y_test: np.array):
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test
        self.max_depth = 5
        self.n_estimators = 100
        
    def createXGBModel(self):
        ''' Creating the XGBoost model
        '''
        xgb_model = xgb.XGBClassifier(objective='reg:squarederror',
                                      learning_rate=0.1, 
                                      n_estimators=self.n_estimators, 
                                      max_depth=self.max_depth)
        return xgb_model
    
    def createGBMModel(self):
        ''' Creating the Gradient Boosting model
        '''
        gbm_model = GradientBoostingClassifier(n_estimators=self.n_estimators, 
                                              learning_rate=0.1,
                                               max_depth=self.max_depth)
        return gbm_model

    def createRFModel(self):
        ''' This will create the Random Forest model
        '''
        rf = RandomForestClassifier(n_estimators=self.n_estimators, 
                                    max_depth=self.max_depth, 
                                    min_samples_split=2, random_state=0)
        return rf
    
    def trainModels(self):
        xgb_model = self.createXGBModel()
        gbm_model = self.createGBMModel()
        rf_model  = self.createRFModel()
        
        xgb_model.fit(self.X_train, self.y_train)
        gbm_model.fit(self.X_train, self.y_train)
        rf_model.fit(self.X_train, self.y_train)
        
        return xgb_model, gbm_model, rf_model
    
    def evalModels(self, xgb_model, gbm_model, rf_model):
        xgb_pred = xgb_model.predict(self.X_test)
        gbm_pred = gbm_model.predict(self.X_test)
        rf_pred  = rf_model.predict(self.X_test)
        
        xgb_confusion = confusion_matrix(self.y_test, xgb_pred)
        gbm_confusion = confusion_matrix(self.y_test, gbm_pred)
        rf_confusion  = confusion_matrix(self.y_test, rf_pred)
        
        conf_matrix = [xgb_confusion, gbm_confusion, rf_confusion]
        
        accuracy = [accuracy_score(self.y_test, xgb_pred),
                    accuracy_score(self.y_test, gbm_pred),
                    accuracy_score(self.y_test, rf_pred)]
        
        prec_score = [precision_score(self.y_test, xgb_pred), 
                      precision_score(self.y_test, gbm_pred),
                      precision_score(self.y_test, rf_pred)]
        
        rec_score = [recall_score(self.y_test, xgb_pred),
                     recall_score(self.y_test, gbm_pred),
                     recall_score(self.y_test, rf_pred)]

        f1score = [f1_score(self.y_test, xgb_pred),
                   f1_score(self.y_test, gbm_pred),
                   f1_score(self.y_test, rf_pred)]
        
        return conf_matrix, accuracy, prec_score, rec_score, f1score

if __name__ == '__main__':    
    # Creaing the training and testing dataset
    prepare_data = prepareData('data/combined_data.csv')
    X_train, X_test, y_train, y_test = prepare_data.createTestTrainSplit()
    
    # Defining the models
    model = model(X_train, X_test, y_train, y_test)
    xgb_model, gbm_model, rf_model = model.trainModels()
    
    # Evaluating the models
    conf_matrix, accuracy, prec_score, rec_score, f1score = model.evalModels(
        xgb_model, gbm_model, rf_model)


