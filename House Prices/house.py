# Artificial Neural Network for house prices
# Isaac Tesla

# Part 1 - Data Preprocessing
# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('train.csv')


#=============================================================================
#=============================================================================
# turn every single column into a number and the standardise.
# start with row 3 - MSZoning
dataset.MSZoning.unique()
# we can see there are 5 categories [RL, RM, C (all), FV, RH]
# we shall number them 1 - 5
dataset.MSZoning.replace(['RL', 'RM', 'C (all)', 'FV', 'RH'], [0, 1, 2, 3, 4], inplace=True)

# next column to check is Street
dataset.Street.unique()
# with only 2 values we will set these as 1 and 0
dataset.Street.replace(['Pave', 'Grvl'], [0, 1], inplace=True)

# next column is Alley
dataset.Alley.unique()
dataset.Alley.replace(['Pave', 'Grvl'], [1, 2], inplace=True)
dataset.Alley = dataset.Alley.fillna(0)

# next column is LotShape
dataset.LotShape.unique()
dataset.LotShape.replace(['Reg', 'IR1', 'IR2', 'IR3'], [0, 1, 2, 3], inplace=True)

# next column is LandContour
dataset.LandContour.unique()
dataset.LandContour.replace(['Lvl', 'Bnk', 'Low', 'HLS'], [0, 1, 2, 3], inplace=True)

# next column is Utilities
dataset.Utilities.unique()
dataset.Utilities.replace(['AllPub', 'NoSeWa'], [0,1], inplace=True)

# next column is LotConfig
dataset.LotConfig.unique()
dataset.LotConfig.replace(['Inside', 'FR2', 'Corner', 'CulDSac', 'FR3'], [0,1,2,3,4], inplace=True)

# next column is LandSlope
dataset.LandSlope.unique()
dataset.LandSlope.replace(['Gtl', 'Mod', 'Sev'], [0, 1, 2], inplace=True)

# next column is Neighbourhood
dataset.Neighborhood.unique()
dataset.Neighborhood.replace(['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst',
       'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes',
       'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert',
       'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU',
       'Blueste'], [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24], inplace=True)

# next column is Condition1
dataset.Condition1.unique()
dataset.Condition1.replace(['Norm', 'Feedr', 'PosN', 'Artery', 'RRAe', 'RRNn', 'RRAn', 'PosA',
       'RRNe'], [0,1,2,3,4,5,6,7,8], inplace=True)

# next column is Condition2
dataset.Condition2.unique()
dataset.Condition2.replace(['Norm', 'Artery', 'RRNn', 'Feedr', 'PosN', 'PosA', 'RRAn', 'RRAe'], [0,1,2,3,4,5,6,7], inplace=True)

# next column is BldgType
dataset.BldgType.unique()
dataset.BldgType.replace(['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs'], [0,1,2,3,4], inplace=True)

# next column is HouseStyle
dataset.HouseStyle.unique()
dataset.HouseStyle.replace(['2Story', '1Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf',
       '2.5Fin'], [0,1,2,3,4,5,6,7], inplace=True)

# next column is RoofStyle
dataset.RoofStyle.unique()
dataset.RoofStyle.replace(['Gable', 'Hip', 'Gambrel', 'Mansard', 'Flat', 'Shed'], [0,1,2,3,4,5], inplace=True)

# next column is RoofMatl
dataset.RoofMatl.unique()
dataset.RoofMatl.replace(['CompShg', 'WdShngl', 'Metal', 'WdShake', 'Membran', 'Tar&Grv',
       'Roll', 'ClyTile'], [0,1,2,3,4,5,6,7], inplace=True)

# next column is Exterior1st
dataset.Exterior1st.unique()
dataset.Exterior1st.replace(['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing',
       'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'BrkComm', 'AsphShn',
       'Stone', 'ImStucc', 'CBlock'], [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], inplace=True)

# next column is Exterior2nd
dataset.Exterior2nd.unique()
dataset.Exterior2nd.replace(['VinylSd', 'MetalSd', 'Wd Shng', 'HdBoard', 'Plywood', 'Wd Sdng',
       'CmentBd', 'BrkFace', 'Stucco', 'AsbShng', 'Brk Cmn', 'ImStucc',
       'AsphShn', 'Stone', 'Other', 'CBlock'], [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], inplace=True)

# next column is MasVnrType
dataset.MasVnrType.unique()
dataset.MasVnrType.replace(['BrkFace', 'None', 'Stone', 'BrkCmn'], [1,2,3,4], inplace=True)
dataset.MasVnrType = dataset.MasVnrType.fillna(0)

# next column ExterQual
dataset.ExterQual.unique()
dataset.ExterQual.replace(['Gd', 'TA', 'Ex', 'Fa'], [0,1,2,3], inplace=True)

# next column ExterCond
dataset.ExterCond.unique()
dataset.ExterCond.replace(['TA', 'Gd', 'Fa', 'Po', 'Ex'], [0,1,2,3,4], inplace=True)

# next column Foundation
dataset.Foundation.unique()
dataset.Foundation.replace(['PConc', 'CBlock', 'BrkTil', 'Wood', 'Slab', 'Stone'], [0,1,2,3,4,5], inplace=True)

# next column BsmtQual
dataset.BsmtQual.unique()
dataset.BsmtQual.replace(['Gd', 'TA', 'Ex', 'Fa'], [1,2,3,4], inplace=True)
dataset.BsmtQual = dataset.BsmtQual.fillna(0)

# next column BsmtCond
dataset.BsmtCond.unique()
dataset.BsmtCond.replace(['TA', 'Gd', 'Fa', 'Po'], [1,2,3,4], inplace=True)
dataset.BsmtCond = dataset.BsmtCond.fillna(0)

# next column BsmtExposure
dataset.BsmtExposure.unique()
dataset.BsmtExposure.replace(['No', 'Gd', 'Mn', 'Av'], [1,2,3,4], inplace=True)
dataset.BsmtExposure = dataset.BsmtExposure.fillna(0)

# next column BsmtFinType1
dataset.BsmtFinType1.unique()
dataset.BsmtFinType1.replace(['GLQ', 'ALQ', 'Unf', 'Rec', 'BLQ', 'LwQ'], [1,2,3,4,5,6], inplace=True)
dataset.BsmtFinType1 = dataset.BsmtFinType1.fillna(0)

# next column BsmtFinType2
dataset.BsmtFinType2.unique()
dataset.BsmtFinType2.replace(['Unf', 'BLQ', 'ALQ', 'Rec', 'LwQ', 'GLQ'], [1,2,3,4,5,6], inplace=True)
dataset.BsmtFinType2 = dataset.BsmtFinType2.fillna(0)

# next column Heating
dataset.Heating.unique()
dataset.Heating.replace(['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor'], [0,1,2,3,4,5], inplace=True)

# next column HeatingQC
dataset.HeatingQC.unique()
dataset.HeatingQC.replace(['Ex', 'Gd', 'TA', 'Fa', 'Po'], [0,1,2,3,4], inplace=True)

# next column CentralAir
dataset.CentralAir.unique()
dataset.CentralAir.replace(['Y', 'N'], [1,0], inplace=True)

# next column Electrical
dataset.Electrical.unique()
dataset.Electrical.replace(['SBrkr', 'FuseF', 'FuseA', 'FuseP', 'Mix'], [1,2,3,4,5], inplace=True)
dataset.Electrical = dataset.Electrical.fillna(0)

# next column KitchenQual
dataset.KitchenQual.unique()
dataset.KitchenQual.replace(['Gd', 'TA', 'Ex', 'Fa'], [0,1,2,3], inplace=True)

# next column Functional
dataset.Functional.unique()
dataset.Functional.replace(['Typ', 'Min1', 'Maj1', 'Min2', 'Mod', 'Maj2', 'Sev'], [0,1,2,3,4,5,6], inplace=True)

# next column FireplaceQu
dataset.FireplaceQu.unique()
dataset.FireplaceQu.replace(['TA', 'Gd', 'Fa', 'Ex', 'Po'], [1,2,3,4,5], inplace=True)
dataset.FireplaceQu = dataset.FireplaceQu.fillna(0)

# next column GarageType
dataset.GarageType.unique()
dataset.GarageType.replace(['Attchd', 'Detchd', 'BuiltIn', 'CarPort','Basment', '2Types'], [1,2,3,4,5,6], inplace=True)
dataset.GarageType = dataset.GarageType.fillna(0)

# next column GarageFinish
dataset.GarageFinish.unique()
dataset.GarageFinish.replace(['RFn', 'Unf', 'Fin'], [1,2,3], inplace=True)
dataset.GarageFinish = dataset.GarageFinish.fillna(0)

# next column GarageQual
dataset.GarageQual.unique()
dataset.GarageQual.replace(['TA', 'Fa', 'Gd', 'Ex', 'Po'], [1,2,3,4,5], inplace=True)
dataset.GarageQual = dataset.GarageQual.fillna(0)

# next column GarageCond
dataset.GarageCond.unique()
dataset.GarageCond.replace(['TA', 'Fa', 'Gd', 'Po', 'Ex'], [1,2,3,4,5], inplace=True)
dataset.GarageCond = dataset.GarageCond.fillna(0)

# next column PavedDrive
dataset.PavedDrive.unique()
dataset.PavedDrive.replace(['Y', 'N', 'P'], [2,0,1], inplace=True)

# next column PoolQC
dataset.PoolQC.unique()
dataset.PoolQC.replace(['Ex', 'Fa', 'Gd'], [3,1,2], inplace=True)
dataset.PoolQC = dataset.PoolQC.fillna(0)

# next column Fence
dataset.Fence.unique()
dataset.Fence.replace(['MnPrv', 'GdWo', 'GdPrv', 'MnWw'], [1,2,3,4], inplace=True)
dataset.Fence = dataset.Fence.fillna(0)

# next column MiscFeature
dataset.MiscFeature.unique()
dataset.MiscFeature.replace(['Shed', 'Gar2', 'Othr', 'TenC'], [1,2,3,4], inplace=True)
dataset.MiscFeature = dataset.MiscFeature.fillna(0)

# next column SaleType
dataset.SaleType.unique()
dataset.SaleType.replace(['WD', 'New', 'COD', 'ConLD', 'ConLI', 'CWD', 'ConLw', 'Con', 'Oth'], [0,1,2,3,4,5,6,7,8], inplace=True)

# next column SaleCondition
dataset.SaleCondition.unique()
dataset.SaleCondition.replace(['Normal', 'Abnorml', 'Partial', 'AdjLand', 'Alloca', 'Family'], [0,1,2,3,4,5], inplace=True)

#==============================================================================
#==============================================================================

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 80].values







# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 40, kernel_initializer = 'uniform', activation = 'relu', input_dim = 79))
# classifier.add(Dropout(p = 0.1))

# Adding the second hidden layer
classifier.add(Dense(units = 40, kernel_initializer = 'uniform', activation = 'relu'))
# classifier.add(Dropout(p = 0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Part 4 - Evaluating, Improving and Tuning the ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 40, kernel_initializer = 'uniform', activation = 'relu', input_dim = 79))
    classifier.add(Dense(units = 40, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_