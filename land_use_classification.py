# # -*- coding: utf-8 -*-
# """
# Created on Wed Nov 22 13:31:30 2023

# @author: tranq
# """

import rasterio as rio
import earthpy.plot as ep
import pandas as pd
from rasterio.plot import reshape_as_image, reshape_as_raster
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

with rio.open('raw_data/ESA WorldCover 2020.tif') as src:
    land_cover = src.read()
    profile = src.profile
    src.close()
    
with rio.open('outputs/ghana_prepped_2020.tif') as src:
    ghana_2020 = src.read()
    profile = src.profile
    src.close()
    
data = np.vstack([ghana_2020, land_cover])

# Separate train and test data geographically
train_set = data[..., 625:]
test_set = data[..., :625]

# TCC of the full area
# ep.plot_rgb(data, rgb = (3,2,1), stretch = True)

# Select smaller study area
study_area = train_set[:, :625, :625]

columns = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9",
           "B11", "B12", "NDVI", "MNDWI", "BSI", "VV", "VH", "M", "class"]

def prep_image_for_ml(image, columns):
    '''
    Parameters
    ----------
    image : 3D numpy array
        Consisting of 19 bands, including the classification.
    
    columns : list
        column names to be given to the dataframe. Must be same length as number of bands + labels
        
    labelled : Boolean
        Whether the image land use is labelled 

    Returns
    -------
    Dataframe without nans.

    '''
    # change from band, x, y to x, y, band
    # then reshape into z columns, where z is the number of bands
    image_array = reshape_as_image(study_area)
    
    # convert all pixels to rows
    ml_prep_array = image_array.reshape(-1, study_area.shape[0])
    
    train_data = pd.DataFrame(data = ml_prep_array, columns = columns)
    
    # Drop nas
    train_data_cleaned = train_data.dropna().reset_index(drop = True)
    
    return train_data_cleaned

train_data = prep_image_for_ml(study_area, columns)

# Separate into X and y
X = train_data.drop(columns = "class")
y = train_data["class"]

# Make y categorical using label encoder
class_encoder = LabelEncoder().fit(y)
y = class_encoder.transform(y)

## pipeline for ML
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', DummyClassifier(strategy="stratified"))
])

# Dummy F1 score is approximately 0.143 - should be relatively easy to beat
metrics = pd.DataFrame(cross_validate(pipe, X, y, cv = 10, scoring = "f1_macro"))

#print("Dummy metrics")
#print(metrics)
#print(metrics['test_score'].mean())

# X still contains around 300,000 samples
# Learning curve suggests that only 20,000 samples are needed for full training
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .2, stratify = y)

params = {
    'model': [LogisticRegression(max_iter = 3000), 
              # Note that RandomForest is sensitive to imbalanced datasets
              RandomForestClassifier(), 
              LinearSVC(), 
              KNeighborsClassifier()]
}

searcher = GridSearchCV(
    estimator=pipe,
    param_grid=params, 
    cv = 10,
    verbose = 10,
    scoring = 'f1_macro',
    n_jobs = -1
).fit(X_train, y_train)

cv_results = pd.DataFrame(searcher.cv_results_)

cv_results
