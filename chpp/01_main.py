# Import the Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Load the Dataset
housing = pd.read_csv("housing.csv")

# 2. Create the Stratified test set
housing['income_cat'] = pd.cut(housing['median_income'],
                                bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)     # We will work on this data
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)       # Set aside the data

# We will work on the copy of the training data
housing = strat_train_set.copy()

# 3. Seperate Features and Labels
housing_labels = housing['median_house_value'].copy()
housing = housing.drop("median_house_value", axis=1)

# print(housing, housing_labels)

# 4. List the Numberical and Categorical Features COLUMNS
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]

# 5. Let's make the pipeline
# for numerical columns
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")), 
    ("scaler", StandardScaler())
])

# for categorical columns
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Construct the full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

# 6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)