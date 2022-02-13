import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns

st.set_page_config(layout='wide')


# -------------------- FUNCTIONS ---------------------

# import data from the url
# with whitespace separator and column names
#
# import_data:
#
#     # DATA PRE-PROCESSING
#     1. Download the data
#     2. One hot encode the origin column
#     3. Drop 'car name' column
#     4. Drop na values
#
@st.cache
def import_data():

    # where to get the data and what to call the columns
    url = 'https://code.datasciencedojo.com/datasciencedojo/datasets/raw/master/Auto%20MPG/auto-mpg.data'
    names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']

    # steps 1-4
    df = pd.read_csv(url, sep='\s+', names=names, na_values='?')
    df = pd.get_dummies(df, columns=['origin'])
    df.drop(columns=['car_name'], inplace=True)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)
    return df




# --------------------- APP SETUP --------------------

df = import_data()




# ---------------------- ST APP ----------------------
st.write('# Auto MPG Efficiency Prediction!')
st.write('Click the checkbox to see dataset detailst. . .')
if st.checkbox('Show rawdata'):
    st.write(df)

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

We can also see the density of each variable. For example, most cars get around 20 mpg and accelerate at about 15m/s^2

The cylinders variable is interesting in that it has multiple peaks. This actually makes sense because number of cylinders is discrete, i.e. we a 4.75 cylinder engine cannot exist.
''')


# PLOTTING ALTERNATIVE: https://altair-viz.github.io/gallery/scatter_matrix.html
fig = sns.pairplot(df[['mpg', 'horsepower', 'acceleration', 'displacement', 'cylinders', 'weight']], diag_kind='kde', palette='flare')
st.write(fig.fig)

st.write('''
We may also plot variables against some value we want to predict. In this case we are examining the relationship between most car features and mpg.

This allows us to see multi-variable correlations. For example, in the bottom left chart we see low weight and low horsepower correlates to low mpg.

We can also see the distribution of each variable with respect to miles per gallon. The top left panel shows the majority of low horsepower cars get better mpg

You can also see that almost all low displacement cars have low mpg.
''')
fig = sns.pairplot(df[['mpg', 'horsepower', 'acceleration', 'displacement', 'cylinders', 'weight']], diag_kind='kde', hue='mpg', palette='flare')
st.write(fig.fig)
























# st.markdown('''
# There are nine columns in the *Auto MPG Efficiency* dataset:
# {}\t-> fuel efficiency measured in miles per gallon (mpg, target value)
# \n{}\t-> number of cylinders in the engine
# \n{}\t-> engine displacement (in cubic inches)
# \n{}\t-> engine horsepower
# \n{}\t-> vehicle weight (in pounds)
# \n{}\t-> time to accelerate from O to 60 mph (in seconds)
# \n{}\t-> model year
# \n{}\t-> origin of car (1: American, 2: European, 3: Japanese)
# \n{}\t-> car name
# '''.format('mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name')
# )
