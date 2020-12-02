import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from keras.layers import Dense, Dropout
from keras.models import Model, Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("lebron_dataset.csv",index_col=0)
df.head()

df = df.astype('float32')

print(df.head())

y = df['Win']
X = df.drop(columns=['Win','Winby'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=42)

def my_DL(epochs=6,batchsize=512):
    model = Sequential()
    model.add(Dense(32,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
    return model

def train_hyper_tune(X,y):
    # create the pre-processing component
    my_scaler = StandardScaler()
    my_imputer = SimpleImputer(strategy="median")
    
    # define classifiers
    ## Classifier 1: Logistic Regression
    clf_LR = LogisticRegression(random_state=0,penalty='elasticnet',solver='saga')
    ## Classifier 2: Random Forest Classifier
    clf_RF = RandomForestClassifier(random_state=0)
    ## Classifier 3: Deep Learning Binary Classifier
    clf_DL = KerasClassifier(build_fn=my_DL)
    
    # define pipeline for three classifiers
    ## clf_LR
    pipe1 = Pipeline([('imputer', my_imputer), ('scaler', my_scaler), ('lr_model',clf_LR)])
    ## clf_RF
    pipe2 = Pipeline([('imputer', my_imputer), ('scaler', my_scaler), ('rf_model',clf_RF)])
    ## clf_DL
    pipe3 = Pipeline([('imputer', my_imputer), ('scaler', my_scaler), ('dl_model',clf_DL)])
    
    # create hyperparameter space of the three models
    ## clf_LR
    param_grid1 = {
        'lr_model__C' : [1e-1,1,10],
        'lr_model__l1_ratio' : [0,0.5,1]
    }
    ## clf_RF
    param_grid2 = {
        'rf_model__n_estimators' : [50,100],
        'rf_model__max_features' : [0.8,"auto"],
        'rf_model__max_depth' : [4,5]
    }
    ## clf_DL
    param_grid3 = {
        'dl_model__epochs' : [6,12,18,24],
        'dl_model__batchsize' : [256,512]
    }
    
    # set GridSearch via 5-fold cross-validation
    ## clf_LR
    grid1 = GridSearchCV(pipe1, cv=5, param_grid=param_grid1)
    ## clf_RF
    grid2 = GridSearchCV(pipe2, cv=5, param_grid=param_grid2)
    ## clf_DL
    grid3 = GridSearchCV(pipe3, cv=5, param_grid=param_grid3)
    
    # run the hyperparameter tunning process
    grid1.fit(X,y)
    grid2.fit(X,y)
    grid3.fit(X,y)
    
    # return results of the tunning process
    return grid1,grid2,grid3,pipe1,pipe2,pipe3

my_grid1,my_grid2,my_grid3,my_pipe1,my_pipe2,my_pipe3 = train_hyper_tune(X_train, y_train)

print(train_hyper_tune(X_train, y_train))
