import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# load dataset
data = pd.read_csv('train.csv')

# understand dataset
print(data.head())
print(data.describe())
print(data.info())
print(data.apply(lambda x: len(x.unique())))

############# Pre-processing ####################3
print(data.isnull().sum())

cat_col = []
for x in data.dtypes.index:
    if data.dtypes[x] == 'object':
        cat_col.append(x)
print(cat_col)

cat_col.remove('Item_Identifier')
cat_col.remove('Outlet_Identifier')
print(cat_col)

for col in cat_col:
    print(col)
    print(data[col].value_counts())

# filling null values
item_weight_mean= data.pivot_table(values='Item_Weight', index= 'Item_Identifier')
print(item_weight_mean)

miss_bool= data['Item_Weight'].isnull()
print(miss_bool)

for i, item in enumerate(data['Item_Identifier']):
    if miss_bool[i]:
        if item in item_weight_mean:
            data['Item_Weight'][i] = item_weight_mean.loc[item]['Item_Weight']
        else:
            data['Item_Weight'][i] = np.mean(data['Item_Weight'])
print(data['Item_Weight'].isnull().sum())

outlet_size_node= data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
print(outlet_size_node)

miss_bool= data['Outlet_Size'].isnull()
outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
print(outlet_size_mode)

miss_bool = data['Outlet_Size'].isnull()
data.loc[miss_bool, 'Outlet_Size'] = data.loc[miss_bool, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
print(data['Outlet_Size'].isnull().sum())
print(sum(data['Item_Visibility']==0))

data.loc[:, 'Item_Visibility'].replace([0], [data['Item_Visibility'].mean()], inplace=True)
print(sum(data['Item_Visibility']==0))

data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})
print(data['Item_Fat_Content'].value_counts())


# Creation of new attribute

data['New_Item_Type'] = data['Item_Identifier'].apply(lambda x: x[:2])
print(data['New_Item_Type'])

data['New_Item_Type'] = data['New_Item_Type'].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})
print(data['New_Item_Type'].value_counts())

data.loc[data['New_Item_Type']=='Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'
print(data['Item_Fat_Content'].value_counts())

data['Outlet_Years'] = 2020 - data['Outlet_Establishment_Year']
print(data['Outlet_Years'])
print(data.head())

########################### Data Analysis #######################
sns.distplot(data['Item_Weight'])
plt.show()
sns.distplot(data['Item_Visibility'])
plt.show()
sns.distplot(data['Item_MRP'])
plt.show()
sns.distplot(data['Item_Outlet_Sales'])
plt.show()
data['Item_Outlet_Sales'] = np.log(1+data['Item_Outlet_Sales'])
sns.distplot(data['Item_Outlet_Sales'])
plt.show()
sns.countplot(data["Item_Fat_Content"])
plt.show()

l = list(data['Item_Type'].unique())
chart = sns.countplot(data["Item_Type"])
print(chart.set_xticklabels(labels=l, rotation=90))
plt.show()
sns.countplot(data['Outlet_Establishment_Year'])
plt.show()
sns.countplot(data['Outlet_Size'])
plt.show()
sns.countplot(data['Outlet_Location_Type'])
plt.show()
sns.countplot(data['Outlet_Type'])
plt.show()


#############Corelation Matrix#################
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

############Label Encoding########################
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
cat_col = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type']
for col in cat_col:
    data[col] = le.fit_transform(data[col])

#Onehot Encoding
data = pd.get_dummies(data, columns=['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type'])

#Input Split
X = data.drop(columns=['Outlet_Establishment_Year', 'Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])
y = data['Item_Outlet_Sales']

######################### Training #################################
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

def train(model, X, y):
    model.fit(X, y)
    pred = model.predict(X)
    cv_score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    cv_score = np.abs(np.mean(cv_score))
    print("Model Report")
    print("MSE:", mean_squared_error(y, pred))
    print("CV Score:", cv_score)
    return cv_score, mean_squared_error(y, pred)

from sklearn.linear_model import Lasso
print()
print('Using Lasso : ')
model = Lasso()
lassoCV, lassoMSE=train(model, X, y)
coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind='bar', title="Model Coefficients")
plt.show()

from sklearn.tree import DecisionTreeRegressor
print()
print('Using Decision Tree Regressor : ')
model = DecisionTreeRegressor()
DT_CV, DT_MSE= train(model, X, y)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title="Feature Importance")
plt.show()

from sklearn.ensemble import RandomForestRegressor
print()
print('Using Random Forest Regressor : ')
model = RandomForestRegressor()
RF_CV, RF_MSE= train(model, X, y)
coef = pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False)
coef.plot(kind='bar', title="Feature Importance")
plt.show()

