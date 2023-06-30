import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error

# Custom transformer to compute ratio features
class RatioFeaturesTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Compute bedrooms_ratio and people_per_house ratio
        X['bedrooms_ratio'] = X['total_bedrooms'] / X['total_rooms']
        X['people_per_house'] = X['population'] / X['households']
        return X

# Load the dataset
df = pd.read_csv('./data/housing.csv')

# Select the features and target variable
features = df.drop(columns=["median_house_value"])
target = df["median_house_value"]

# Define the numerical and categorical features
numerical_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']

# Create the transformations for numerical and categorical features
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Create the column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer([
    ('numerical_transformer', numerical_transformer, numerical_features),
    ('ratio_transformer', RatioFeaturesTransformer(), numerical_features)  # Apply custom ratio features transformer
])

# Create the pipeline by combining the preprocessor with the linear regression model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression ())
])

# Preprocess the data
X_preprocessed = preprocessor.fit_transform(features)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test data
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)