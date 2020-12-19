#!/usr/bin/env python
# coding: utf-8

# ## Dependencies

# In[233]:


# Data Management/Investigation
import pandas as pd
from pandas.api.types import CategoricalDtype # Ordering categories
import numpy as np
import missingno as miss

# Plotting libraries
from plotnine import *
import matplotlib.pyplot as plt

# For pre-processing data 
from sklearn import preprocessing as pp 
from sklearn.compose import ColumnTransformer 

# For splits and CV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold # Cross validation 
from sklearn.model_selection import cross_validate # Cross validation 
from sklearn.model_selection import GridSearchCV # Cross validation + param. tuning.

# Machine learning methods 
from sklearn.linear_model import LinearRegression as LM
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import tree # For plotting the decision tree rules
from sklearn.ensemble import BaggingRegressor as Bag


# For evaluating our model's performance
import sklearn.metrics as m

# Pipeline to combine modeling elements
from sklearn.pipeline import Pipeline

# Misc
import warnings
warnings.filterwarnings("ignore")

#lab encoder
from sklearn import preprocessing
from sklearn import utils


# In[181]:


# Set seed
np.random.seed(123)
# Data import
game = pd.read_csv('video_game.csv')


# In[119]:


game.head()


# In[113]:


len(game)


# ## Data Overall Review

# In[10]:


# Plot the continuous Variables 
d = game.select_dtypes(include="float64").melt()
(
    ggplot(d,aes(x="value")) +
    geom_histogram(bins=25) +
    facet_wrap("variable",scales='free') +
    theme(figure_size=(10,3),
          subplots_adjust={'wspace':0.25})
)


# In[95]:


#Change categorical value to numerical
for col in ['Publisher_x','Genre','Platform']:
    game[col] = game[col].astype('category')


# In[13]:


d = game.select_dtypes(include="category").melt()
(
    ggplot(d,aes(x="value")) +
    geom_bar() +
    facet_wrap("variable",scales='free') +
    theme(figure_size=(15,7),
          subplots_adjust={'wspace':0.25,
                           'hspace':0.75},
         axis_text_x=element_text(rotation=45, hjust=1))
)


# ## Data proprocessing

# #### convert categorical to numberical

# - Genre

# In[182]:


game.Genre.value_counts()


# In[183]:


game['Genre'] = game['Genre'].replace(['Action'],11)
game['Genre'] = game['Genre'].replace(['Sports'],10)
game['Genre'] = game['Genre'].replace(['Shooter'],9)
game['Genre'] = game['Genre'].replace(['Platform'],8)
game['Genre'] = game['Genre'].replace(['Role-Playing'],7)
game['Genre'] = game['Genre'].replace(['Misc'],6)
game['Genre'] = game['Genre'].replace(['Racing'],5)
game['Genre'] = game['Genre'].replace(['Fighting'],4)
game['Genre'] = game['Genre'].replace(['Simulation'],3)
game['Genre'] = game['Genre'].replace(['Adventure'],2)
game['Genre'] = game['Genre'].replace(['Puzzle'],1)
game['Genre'] = game['Genre'].replace(['Strategy'],0)


# In[184]:


game.head()


# In[185]:


game.Genre.value_counts()


# - Platform

# In[188]:


game.Platform.value_counts()


# In[189]:


game['platform'] = 0


# In[190]:


game.loc[game['Platform'] == 'PS3', 'platform'] = 5
game.loc[game['Platform'] == 'X360', 'platform'] = 4
game.loc[game['Platform'] == 'PS2', 'platform'] = 3
game.loc[game['Platform'] == 'Wii', 'platform'] = 2
game.loc[game['Platform'] == 'DS', 'platform'] = 1


# In[191]:


game.head()


# In[234]:


# Drop platform
game = game.drop(columns=["Platform"])


# In[193]:


game.platform.value_counts()


# - Publisher_x

# In[198]:


game.Publisher_x.value_counts()[0:10]


# In[199]:


game['publisher'] = 0


# In[200]:


game.loc[game['Publisher_x'] == 'Electronic Arts', 'publisher'] = 7
game.loc[game['Publisher_x'] == 'Activision', 'publisher'] = 6
game.loc[game['Publisher_x'] == 'Ubisoft', 'publisher'] = 5
game.loc[game['Publisher_x'] == 'THQ', 'publisher'] = 4
game.loc[game['Publisher_x'] == 'Sega', 'publisher'] = 3
game.loc[game['Publisher_x'] == 'Take-Two Interactive', 'publisher'] = 2
game.loc[game['Publisher_x'] == 'Konami Digital Entertainment', 'publisher'] = 1


# In[201]:


game.head()


# In[202]:


game.publisher.value_counts()


# In[203]:


# Drop publisher
game = game.drop(columns=["Publisher_x"])


# - volumn sales

# In[204]:


game['NA_Sales'] = np.log(game['NA_Sales'] + 1)
game['EU_Sales'] = np.log(game['EU_Sales'] + 1)
game['JP_Sales'] = np.log(game['JP_Sales'] + 1)
game['Other_Sales'] = np.log(game['Other_Sales'] + 1)
game['Global_Sales'] = np.log(game['Global_Sales'] + 1)


# In[205]:


game.head()


# ## Split Model

# In[230]:


y = game['Global_Sales']
X = game[['Year','NA_Sales',"Critic Score","platform","publisher","Rev"]]


# In[231]:


train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=.25,random_state=123)


# In[232]:


D = train_X.copy()
D['Global_Sales'] = train_y

(
    ggplot(D.melt(id_vars=["Global_Sales"]),
           aes(x="value",y="Global_Sales"))+
    geom_point(alpha=.5) +
    facet_wrap("variable",scales="free") +
    geom_smooth(method="lm",se=False,color="red") +
    theme_minimal() +
    theme(figure_size = (10,3)) 
)


# ## Modeling

# In[243]:


from sklearn import preprocessing
from sklearn import utils


# In[258]:


lab_enc = preprocessing.LabelEncoder()
train_y = lab_enc.fit_transform(train_y)


# In[259]:


print(utils.multiclass.type_of_target(train_y))


# In[260]:


fold_generator = KFold(n_splits=5, shuffle=True,random_state=111)


# In[261]:


## use mse to score the model
use_metrics = ["neg_mean_squared_error"]


# - Linear Model

# In[262]:


lm_scores = cross_validate(LM(),train_X,train_y, cv = fold_generator, scoring =use_metrics)


# - KNN

# In[263]:


knn_scores = cross_validate(KNN(),train_X,train_y, cv = fold_generator, scoring =use_metrics)


# - Decision Tree

# In[264]:


dt_scores = cross_validate(DT(),train_X,train_y, cv = fold_generator, scoring =use_metrics)


# In[270]:


col_names = list(train_X)


# In[271]:


mod = DT(max_depth=3) 
mod.fit(train_X,train_y) 

# Plot the tree
plt.figure(figsize=(12,8),dpi=300)
rules = tree.plot_tree(mod,feature_names = col_names,fontsize=8)


# - Bagging

# In[394]:


bag_scores = cross_validate(Bag(),train_X,train_y, cv = fold_generator, scoring =use_metrics)


# - Random Forest

# In[268]:


rf_scores = cross_validate(RF(),train_X,train_y, cv = fold_generator, scoring =use_metrics)


# #### Compute scores

# In[269]:


# Collect all the metrics we care about as a dictionary 
collect_scores = dict(lm = lm_scores['test_neg_mean_squared_error']*-1,
     knn = knn_scores['test_neg_mean_squared_error']*-1,
     dt = dt_scores['test_neg_mean_squared_error']*-1,
     bag = bag_scores['test_neg_mean_squared_error']*-1,
     rf = rf_scores['test_neg_mean_squared_error']*-1)

# Convert to a data frame and reshape
collect_scores = pd.DataFrame(collect_scores).melt(var_name="Model",value_name="MSE")
collect_scores


# In[382]:



# Get the order of the models
order = (collect_scores.groupby('Model').mean().sort_values(by="MSE").index.tolist())

# Plot
myPlot =(
    ggplot(collect_scores,
          aes(x="Model",y="MSE")) +
    geom_boxplot() +
    scale_x_discrete(limits=order) +
    labs(x="Model",y="Mean Squared Error") +
    coord_flip() +
    theme_minimal() +
    theme(dpi=150)
)
ggsave(filename="notune.png", plot=myPlot)


# ## Tuning Hypothesis model

# ### GridSearch

# - KNN

# In[281]:


mod = KNN() # Initialize the model class
mod.get_params() # report all the available tunning parameters


# In[282]:


knn_tune_params = {'n_neighbors':[1, 10, 25, 35, 50, 75, 100, 250]}


# In[283]:



tune_knn = GridSearchCV(KNN(),knn_tune_params,
                        cv = fold_generator,
                        scoring='neg_mean_squared_error',
                        n_jobs=4)


# In[284]:


tune_knn.fit(train_X,train_y)


# In[285]:


tune_knn.best_params_


# In[286]:


tune_knn.best_score_


# - Decision Tress

# In[288]:


DT().get_params()


# In[290]:


tune_dt = GridSearchCV(DT(),{'max_depth':[i for i in range(10)]},
                        cv = fold_generator,
                        scoring='neg_mean_squared_error',
                        n_jobs=4)


# In[291]:


tune_dt.fit(train_X,train_y)


# In[292]:


tune_dt.best_params_


# In[293]:


tune_dt.best_score_


# In[294]:


RF().get_params()


# In[295]:


rf_params = {'max_depth':[1,2,3],
             'n_estimators':[100,500,1000],
              'max_features': [1,2]} # Only have three total. 
tune_rf = GridSearchCV(RF(),rf_params,
                        cv = fold_generator,
                        scoring='neg_mean_squared_error',
                        n_jobs=4)


# In[296]:


tune_rf.fit(train_X,train_y) 


# In[297]:


tune_rf.best_params_


# In[298]:


tune_rf.best_score_


# In[340]:


# (0) Split the data 
train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=.25,random_state=123)
lab_enc = preprocessing.LabelEncoder()
train_y = lab_enc.fit_transform(train_y)
# (1) Set the folds index to ensure comparable samples
fold_generator = KFold(n_splits=5, shuffle=True,random_state=111)

# (2) Next specify the preprocessing steps
preprocess = ColumnTransformer(transformers=[('num', pp.MinMaxScaler(), ['Year','NA_Sales',"Critic Score","platform","publisher","Rev"])])


# (3) Next Let's create our model pipe (note for the model we leave none as a placeholder)
pipe = Pipeline(steps=[('pre_process', preprocess),
                       ('model',None)])


# (4) Specify the models and their repsective tuning parameters. 
# Note the naming convention here to reference the model key
search_space = [
    # Linear Model
    {'model' : [LM()]},
    
    # KNN with K tuning param
    {'model' : [KNN()],
     'model__n_neighbors':[1]},
    
    # Decision Tree with the Max Depth Param
    {'model': [DT()],
     'model__max_depth':[9]},
    
    # The Bagging decision tree model 
    {'model': [Bag()]},
    
    # Random forest with the N Estimators tuning param
    {'model' : [RF()],
     'model__max_depth':[3],
     'model__n_estimators':[100]},
]


# (5) Put it all together in the grid search
search = GridSearchCV(pipe, search_space, 
                      cv = fold_generator,
                      scoring='neg_mean_squared_error',
                      n_jobs=4)

# (6) Fit the model to the training data
search.fit(train_X,train_y)


# In[395]:


search.best_params_


# In[396]:


bag_mod = search.best_estimator_


# ## Test performance

# In[397]:


pred_y = search.predict(test_X)


# In[398]:


m.mean_squared_error(test_y,pred_y)


# In[399]:


m.r2_score(test_y,pred_y)


# In[384]:


myPlot = (
    ggplot(pd.DataFrame(dict(pred=pred_y,truth=test_y)),
          aes(x='pred',y="truth")) +
    geom_point(alpha=.75) +
    geom_abline(linetype="dashed",color="darkred",size=1) +
    theme_bw() +
    theme(figure_size=(10,7))
)
ggsave(filename="test_performance.png", plot=myPlot)


# ## Model Interpretation

# In[347]:


from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence


# In[348]:


from sklearn.inspection import permutation_importance


# In[349]:


vi = permutation_importance(bag_mod,train_X,train_y,n_repeats=25)


# In[350]:


# Organize as a data frame 
vi_dat = pd.DataFrame(dict(variable=train_X.columns,
                           vi = vi['importances_mean'],
                           std = vi['importances_std']))

# Generate intervals
vi_dat['low'] = vi_dat['vi'] - 2*vi_dat['std']
vi_dat['high'] = vi_dat['vi'] + 2*vi_dat['std']

# But in order from most to least important
vi_dat = vi_dat.sort_values(by="vi",ascending=False).reset_index(drop=True)


vi_dat


# In[387]:


# Plot
myPlot=(
    ggplot(vi_dat,
          aes(x="variable",y="vi")) +
    geom_col(alpha=.5) +
    geom_point() +
    geom_errorbar(aes(ymin="low",ymax="high"),width=.2) +
    theme_bw() +
    scale_x_discrete(limits=vi_dat.variable.tolist()) +
    coord_flip() +
    labs(y="Reduction in AUC ROC",x="")
)
ggsave(filename="vi.png", plot=myPlot)


# ## Partial Dependency Plots

# In[388]:


# Target specific features
features = ['Year','NA_Sales',"Critic Score","platform","publisher","Rev"]

# Calculate the partial dependency
fig, ax = plt.subplots(figsize=(12, 4))
display = plot_partial_dependence(
    bag_mod, train_X, features,n_cols=4,
    n_jobs=4, grid_resolution=30,ax=ax
)
plt.savefig('pdp.png')


# In[389]:


# Feed in the ineraction as a nested list
interacted_features = [['Year','NA_Sales'],['Year',"platform"],['NA_Sales',"platform"]] 

# Then business as usual when plotting
fig, ax = plt.subplots(figsize=(12, 4))
display = plot_partial_dependence(
    bag_mod, train_X, interacted_features,
    n_cols=3,n_jobs=4, grid_resolution=10,ax=ax
)
plt.savefig("pdp2")


# In[361]:


from pdpbox import pdp


# In[366]:


pdp_dist = pdp.pdp_isolate(model = bag_mod, 
                           dataset = train_X,
                           model_features = train_X.columns.tolist(),
                           feature="NA_Sales")


# In[390]:


fig,ax = pdp.pdp_plot(pdp_dist,'NA_Sales',plot_pts_dist=True,center=False,)
pdp.plt.savefig('pdp_NA_Sales', dpi=200)


# ## ICE Plots

# In[376]:


def gen_ice_plot(var_name = "NA_Sales"):
    pdp_dist = pdp.pdp_isolate(model = bag_mod, 
                               dataset = train_X,
                               model_features = train_X.columns.tolist(),
                               feature=var_name)

    fig,ax = pdp.pdp_plot(pdp_dist,var_name,plot_pts_dist=True,
                          center=True,plot_lines=True)
    


# In[391]:


gen_ice_plot()
pdp.plt.savefig('ice_NA_Sales', dpi=200)


# In[392]:


gen_ice_plot(var_name = "platform")
pdp.plt.savefig('ice_platform', dpi=200)


# In[393]:


gen_ice_plot(var_name = "Year")
pdp.plt.savefig('ice_Year', dpi=200)


# In[ ]:




