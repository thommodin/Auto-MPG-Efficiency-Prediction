import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import json
import os

st.set_page_config(layout='wide')


# -------------------- FUNCTIONS ---------------------

# import data from the url
# with whitespace separator and column names (the csv delimitters are quite broken. . .)
#
# import_data:
#
#     # DATA PRE-PROCESSING
#     1. Load the data
#     2. Extract make from car_name column
#     3. Drop 'car_name' column
#     4. One hot encode the origin and make
#     5. Drop na values and reset index
#
@st.cache
def import_data():


    # where to get the data and what to call the columns
    url = 'https://code.datasciencedojo.com/datasciencedojo/datasets/raw/master/Auto%20MPG/auto-mpg.data'
    names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']

    # download and read the data
    df = pd.read_csv(url, sep='\s+', names=names, na_values='?')

    # steps 2-5
    # df['make'] = df['car_name'].apply(lambda x: x.split(' ')[0])
    df.drop(columns=['car_name'], inplace=True)
    df = pd.get_dummies(df, columns=['origin']) #, 'make'])
    df.dropna(inplace=True)
    df = df.sample(frac=1)
    df.reset_index(drop=True, inplace=True)
    return df

# pair_plotting:
#
#     # DATA PAIR-PLOTTING
#     1. Pair plot all
#     2. Pair plot w.r.t. mpg

# --------------------- APP SETUP --------------------

df = import_data()



# ---------------------- ST APP ----------------------
st.write('# Auto MPG Efficiency Prediction!')
st.write('Click the checkbox to see dataset detailst. . .')
if st.checkbox('Show rawdata', value=True):
    st.write(df)



# run the cached pair plots


st.write('''
## Auto MPG Efficiency dataset statistics

We can get a sense for the dataset by examining simple statistics.

For example, we are examining cars from 1970-1982, the number of cylinders varies from as low as 3 to as many as 8 and horsepower ranges from 46 to 230

Note all values are US Units of measurements (pounds, cubic inches and so on)
''')
st.write(df[['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']].describe().T.drop(columns='count'))

st.write('''
## Pair-Plotting variables

We can also get a feeling for which variables might predict others.

For example, in the bottom left corner we can see that weight is inversely proportional to mpg i.e. low weight ∝ high mpg and high weight ∝ low mpg

Some correlations are less prominent. For example, low acceleration is strongly proportional to mpg and middle to high acceleration is somewhat proportional to mpg.

We can also see the density of each variable. For example, most cars get around 20 mpg and accelerate at about 15m/s^2.

The cylinders variable is interesting in that it has multiple peaks. This actually makes sense because number of cylinders is discrete, i.e. we a 4.75 cylinder engine cannot exist.
''')
if st.checkbox('Show all pairplot'):
    st.write(sns.pairplot(df[['mpg', 'horsepower', 'acceleration', 'displacement', 'cylinders', 'weight']], diag_kind='kde', palette='flare').fig)

st.write('''
We may also plot variables against some value we want to predict. In this case we are examining the relationship between most car features and mpg.

This allows us to see multi-variable correlations. For example, in the bottom left chart we see low weight and low horsepower correlates to low mpg.

We can also see the distribution of each variable with respect to miles per gallon. The top left panel shows the majority of low horsepower cars get better mpg.

You can also see that almost all low displacement cars have low mpg.
''')
if st.checkbox('Show pairplot w.r.t mpg'):
    st.write(sns.pairplot(df[['mpg', 'horsepower', 'acceleration', 'displacement', 'cylinders', 'weight']], diag_kind='kde', hue='mpg', palette='flare').fig)


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import multiprocessing



# find number of cores for the system and halve it
# if you are comfortable with it, bump the job count up!
threads = int(max(multiprocessing.cpu_count()/2, 1))

y = df['mpg']
X = df.drop(columns=['mpg'])




adaClassifierSearch = st.button('Start Classifier Search (AdaBoost)')
if adaClassifierSearch:

    # our pipeline for prediction looks like:
    #   1. dimensionality reduction
    #   2. scaling
    #   4. optimizing AdaBoost with CV
    scaler = StandardScaler()
    pca = PCA()
    regressor = AdaBoostRegressor(random_state=42)
    pipeline = make_pipeline(scaler, pca, regressor)

    # we are going to try and optimize some parameters:
    #   1. number of dimensions
    #   2. the number of estimators
    #   3. the type of loss
    params = {
        'pca__n_components':[5, 7, 9, 11],
        'adaboostregressor__n_estimators':[10, 25, 50, 75, 100, 150, 200],
        'adaboostregressor__loss':['linear', 'square', 'exponential']
    }

    # set the search and fit!
    st.write(f'\n\nStarted gridsearch with the following params and {threads} thread/s')
    search = GridSearchCV(pipeline, params, n_jobs=threads, return_train_score=True, verbose=3, cv=10, refit=True, scoring='neg_mean_squared_error')
    if st.checkbox('Show params'):
        st.write(search.get_params())

    search.fit(X, y)
    optimizedAdaBoostRegressor = search.best_estimator_
    st.write(search.best_params_)

    scores = cross_val_score(optimizedAdaBoostRegressor, X, y, cv=10)
    st.write(pd.DataFrame(scores).describe().T)

    st.write(mean_squared_error(optimizedAdaBoostRegressor.predict(X), y))


lassoClassifierSearch = st.button('Start Classifier Search (Lasso)')

if lassoClassifierSearch:

    # our pipeline for prediction looks like:
    #   1. dimensionality reduction
    #   2. scaling
    #   4. optimizing AdaBoost with CV
    scaler = StandardScaler()
    pca = PCA()
    regressor = Lasso(random_state=42)
    pipeline = make_pipeline(scaler, pca, regressor)

    # we are going to try and optimize some parameters:
    #   1. number of dimensions
    #   2. the number of estimators
    #   3. the type of loss
    params = {
        'pca__n_components':[5, 7, 9, 11],
        'lasso__alpha':[0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0]
    }

    # set the search and fit!
    st.write(f'\n\nStarted gridsearch with the following params and {threads} thread/s')
    search = GridSearchCV(pipeline, params, n_jobs=threads, return_train_score=True, verbose=3, cv=10, refit=True, scoring='neg_mean_squared_error')
    if st.checkbox('Show params'):
        st.write(search.get_params())

    search.fit(X, y)
    optimizedLassoRegressor = search.best_estimator_
    st.write(search.best_params_)

    scores = cross_val_score(optimizedLassoRegressor, X, y, cv=10)
    st.write(pd.DataFrame(scores).describe().T)

    st.write(mean_squared_error(optimizedLassoRegressor.predict(X), y))
