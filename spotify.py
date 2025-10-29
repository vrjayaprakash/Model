import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('spotify.csv')
df.head()

df.describe(include = 'object')
df.isnull().sum().sort_values(ascending = False)
df.info()

print(f"shape of dataset is {df.shape}")
print("\n====================================================\n")
print(f"columns in dataset are {df.columns}")
print("\n====================================================\n")
print(f"information about dataset is \n {df.info()}")
print("\n====================================================\n")
print(f"data distribution of every numerical features \n {df.describe(include = 'int64')}")
print("\n====================================================\n")
print(f"data distribution of every categorical features \n{df.describe(include = 'float64')}")
print("\n====================================================\n")
print(f"number of null values in evry column \n {df.isnull().sum().sort_values(ascending = False)}")

for i in df.columns:
    print(f"unique catagories in {i}")
    print(df[i].value_counts())
    print("\n====================================================\n")
    
cat_col = ["is_churned", "offline_listening", "device_type",
           "subscription_type", "country", "gender"]

cat_num = ["ads_listened_per_week", "age"]

num = df.drop(columns = cat_col + cat_num)

for col in cat_col:
    print(f"churn data distribution for {col}")
    print(df.groupby(col)['is_churned'].value_counts())
    
    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    sns.countplot(x = col, data = df, palette = "Set2", hue = 'is_churned', ax = ax[0])
    ax[0].set_title(f"count plot for {col}")
    
    ax[1].pie(df[col].value_counts(), labels = df[col].value_counts().index, autopct = "%0.01f%%")
    ax[1].set_title(f"Distribution of {col}")
    plt.show()
    
    
for col in cat_num:
    print(f"churn data distribution for {col}")
    print(df[col].value_counts())
    
    fig, ax = plt.subplots(2, 1, figsize = (12, 12))
    sns.countplot(x = col, data = df, palette = "Set2", ax = ax[0])
    ax[0].set_title(f"count plot for {col}")
    
    sns.histplot(x = col, data = df, color = "orange", edgecolor = 'black', kde = True)
    ax[1].set_title(f"Distribution of {col}")
    plt.show()
    
for col in num:
    print(f"churn data distribution for {col}")
    print(df[col].value_counts())
    
    fig, ax = plt.subplots(1, 2, figsize = (12, 6))
    sns.boxplot(x = col, data = df, palette = "Set2", ax = ax[0])
    ax[0].set_title(f"box plot for {col}")
    
    sns.histplot(x = col, data = df, color = 'orange', edgecolor = 'black', kde = True)
    ax[1].set_title(f"Distribution of {col}")
    plt.show()
    
corr = df.select_dtypes(include = [np.number]).corr()

plt.figure(figsize = (10, 6))
sns.heatmap(corr, annot = True, cmap = 'coolwarm', fmt = '.2f')
plt.title("Correlation Matrix")
plt.show()

imp_fea = corr['is_churned'].sort_values(ascending = False)[1:]
print(imp_fea)
sns.barplot(x = imp_fea.values, y = imp_fea.index, palette = 'Set1')
plt.title("Feature Correlation with Target(is_churned)")
plt.xlabel("Correlation Value")
plt.ylabel("Features")
plt.show()

cat_col = ['offline_listening', 'device_type', 'subscription_type', 
           'gender', 'country']
df = pd.get_dummies(df, columns = cat_col, drop_first = True, dtype = int)

X = df.drop(columns = ['is_churned'])
y = df['is_churned']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1 )

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier

logr = LogisticRegression()
knn = KNeighborsClassifier()
svc = SVC()
tree = DecisionTreeClassifier()
forest = RandomForestClassifier()
boost = GradientBoostingClassifier()
ada = AdaBoostClassifier()
bag = BaggingClassifier()

models = [logr, knn, svc, tree, forest, boost, ada, bag]

from sklearn.metrics import accuracy_score
for model in models:
    model.fit(X_train, y_train)
    print(f"{model} is trained successfully")
    print(f" Accuracy score for {model} is {accuracy_score(y_test, model.predict(X_test))}")
    print("\n====================================================\n")
    

from sklearn.model_selection import GridSearchCV

param_grid = {
    "LogisticRegression" : (
        LogisticRegression(max_iter = 1000),
        {"C" : [0.01, 0.1, 1, 10], 'solver' : ['liblinear', 'lbfgs']}
),
    
    "KneighborsClassifier" : (
        KNeighborsClassifier(),
        {"n_neighbors" : [3, 5, 7, 9], 'weights' : ['uniform', 'distance']}
    ),
    
    "SVC" : (
        SVC(),
        {"C" : [0.1, 1, 10], 'kernel' : ['linear', 'rbf', 'poly'], 'gamma' : ['scale', 'auto']}
    ),
    
    "DecisionTreeClassifier" : (
        DecisionTreeClassifier(),
        {"max_depth" : [3, 5, 7, None], 'criterion' : ['gini', 'entropy']}
    ),
    
    "RandomForestClassifier" : (
        RandomForestClassifier(),
        {"n_estimators" : [50, 100, 200],
         "max_depth" : [None, 5, 10],
         "min_samples_split" : [2, 5]
         }
    ),
    
    "GradientBoostingClassifier" : (
        GradientBoostingClassifier(),
        {"n_estimators" : [50, 100, 200],
         "learning_rate" : [0.01, 0.1, 0.2],
         'max_depth' : [3, 5, 7]}
    ),
    
    "AdaBoostclassifier" : (
        AdaBoostClassifier(),
        {'n_estimators' : [50, 100, 200],
         'learning_rate' : [0.01, 0.1, 1]}
        ),
    
    "BaggingClassifier" : (
        BaggingClassifier(),
        {'n_estimators' : [10, 50, 100],
        'max_samples' : [0.5, 0.7, 1.0],
        'max_features' : [0.5, 0.7, 1.0]}
    )
}

results = []

for name, (model, params) in param_grid.items():
    print(f"🔍 Tuning {name}...")
    grid_search = GridSearchCV(model, params, cv = 5, scoring = 'accuracy', n_jobs = -1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    test_score = grid_search.score(X_test, y_test)
    
    print(f"{name} : Best Params : {grid_search.best_params_}")
    print(f"Cv Score = {best_score : .4f}, Test Score = {test_score : .4f}")
    print("\n===================================================================\n")
    
    results.append((name, grid_search.best_params_, best_score, test_score))
 
print(grid_search.best_estimator_)   

import tensorflow as tf
import KerasClassifier as kc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


dim = X_train.shape[1]
model1 = Sequential()
optimizer = Adam(learning_rate = 0.001)

model1.add(Dense(256, input_dim = dim, activation = 'relu' ))
model1.add(Dropout(0.6))
model1.add(Dense(128, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001)))
model1.add(Dropout(0.6))
model1.add(Dense(1, activation = 'sigmoid'))
model1.summary()

model1.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
early_stop = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
history = model1.fit(X_train, y_train, epochs = 150, batch_size = 64, validation_split = 0.2, callbacks = [early_stop])

plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()


    

    

    

    

    
