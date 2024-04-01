# ABOUT DATASET
# 1. pH value: PH is an important parameter in evaluating the acid–base balance of water. It is also the indicator of acidic or alkaline condition of water status. WHO has recommended maximum permissible limit of pH from 6.5 to 8.5. The current investigation ranges were 6.52–6.83 which are in the range of WHO standards.
# 2. Hardness: Hardness is mainly caused by calcium and magnesium salts. These salts are dissolved from geologic deposits through which water travels. The length of time water is in contact with hardness producing material helps determine how much hardness there is in raw water. Hardness was originally defined as the capacity of water to precipitate soap caused by Calcium and Magnesium.
# 3. Solids (Total dissolved solids - TDS): Water has the ability to dissolve a wide range of inorganic and some organic minerals or salts such as potassium, calcium, sodium, bicarbonates, chlorides, magnesium, sulfates etc. These minerals produced un-wanted taste and diluted color in appearance of water. This is the important parameter for the use of water. The water with high TDS value indicates that water is highly mineralized. Desirable limit for TDS is 500 mg/l and maximum limit is 1000 mg/l which prescribed for drinking purpose.
# 4. Chloramines: Chlorine and chloramine are the major disinfectants used in public water systems. Chloramines are most commonly formed when ammonia is added to chlorine to treat drinking water. Chlorine levels up to 4 milligrams per liter (mg/L or 4 parts per million (ppm)) are considered safe in drinking water.
# 5. Sulfate: Sulfates are naturally occurring substances that are found in minerals, soil, and rocks. They are present in ambient air, groundwater, plants, and food. The principal commercial use of sulfate is in the chemical industry. Sulfate concentration in seawater is about 2,700 milligrams per liter (mg/L). It ranges from 3 to 30 mg/L in most freshwater supplies, although much higher concentrations (1000 mg/L) are found in some geographic locations.
# 6. Conductivity: Pure water is not a good conductor of electric current rather’s a good insulator. Increase in ions concentration enhances the electrical conductivity of water. Generally, the amount of dissolved solids in water determines the electrical conductivity. Electrical conductivity (EC) actually measures the ionic process of a solution that enables it to transmit current. According to WHO standards, EC value should not exceeded 400 μS/cm.
# 7. Organic_carbon: Total Organic Carbon (TOC) in source waters comes from decaying natural organic matter (NOM) as well as synthetic sources. TOC is a measure of the total amount of carbon in organic compounds in pure water. According to US EPA < 2 mg/L as TOC in treated / drinking water, and < 4 mg/Lit in source water which is use for treatment.
# 8. Trihalomethanes: THMs are chemicals which may be found in water treated with chlorine. The concentration of THMs in drinking water varies according to the level of organic material in the water, the amount of chlorine required to treat the water, and the temperature of the water that is being treated. THM levels up to 80 ppm is considered safe in drinking water.
# 9. Turbidity: The turbidity of water depends on the quantity of solid matter present in the suspended state. It is a measure of light emitting properties of water and the test is used to indicate the quality of waste discharge with respect to colloidal matter. The mean turbidity value obtained for Wondo Genet Campus (0.98 NTU) is lower than the WHO recommended value of 5.00 NTU.
# 10. Potability: Indicates if water is safe for human consumption where 1 means Potable and 0 means Not potable.
#######################################################################################################################
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import warnings

from xgboost.testing.data import joblib

warnings.simplefilter(action="ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
#######################################################################################################################
df = pd.read_csv("dataset/water_potability.csv")
#######################################################################################################################
# 1- EXPLORATORY DATA ANALYSIS

# general picture
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("##################### Columns #####################")
    print(dataframe.columns)
    print("##################### Info #####################")
    print(dataframe.info())
    print("##################### Number of Unique #####################")
    print(dataframe.nunique())

check_df(df)

# capturing categorical and numerical variables
def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# analysis of categorical variables
def cat_summary(dataframe, col_name, plot=True):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col)

# analysis of numerical variables
def num_summary(dataframe, numerical_col, plot=True):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)


#######################################################################################################################
# 2- FEATURE ENGINEERING

# missing value analysis
df.isnull().sum()

df['ph']=df['ph'].fillna(df.groupby(['Potability'])['ph'].transform('mean'))
df['Sulfate']=df['Sulfate'].fillna(df.groupby(['Potability'])['Sulfate'].transform('mean'))
df['Trihalomethanes']=df['Trihalomethanes'].fillna(df.groupby(['Potability'])['Trihalomethanes'].transform('mean'))


# outlier analysis
df.describe().T

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)


# correlation
df[num_cols].corr()
df.corr()

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(train_df[train_num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)


#######################################################################################################################
# 3- MODELLING

y = df["Potability"] # bağımlı değişken
X = df.drop(["Potability"], axis=1) # bağımsız değişkenler


# scaling
scaled = StandardScaler().fit_transform(df[num_cols]) # numpy array'i
df[num_cols] = pd.DataFrame(scaled, columns=df[num_cols].columns) # pandas dataframe'i


def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier(force_row_wise=True, force_col_wise=True),
                   #('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=10, scoring=scoring, n_jobs=-1)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

# classification_report(base_models(X,y))

base_models(X, y, scoring="roc_auc")
# roc_auc: 0.5031 (LR)
# roc_auc: 0.716 (CART)
# roc_auc: 0.8472 (RF)
# roc_auc: 0.587 (KNN)
# roc_auc: 0.6482 (SVC)
# roc_auc: 0.8092 (Adaboost)
# roc_auc: 0.8544 (GBM)
# roc_auc: 0.8445 (XGBoost)
# roc_auc: 0.856 (LightGBM)

# LightGBM > GBM > RF > XGBoost > Ababoost

base_models(X, y, scoring="f1")
# f1: 0.0016 (LR)
# f1: 0.6497 (CART)
# f1: 0.6812 (RF)
# f1: 0.4189 (KNN)
# f1: 0.3602 (SVC)
# f1: 0.5853 (Adaboost)
# f1: 0.6601 (GBM)
# f1: 0.6721 (XGBoost)
# f1: 0.682 (LightGBM)

base_models(X, y, scoring="precision")
# precision: 0.1 (LR)
# precision: 0.6512 (CART)
# precision: 0.7621 (RF)
# precision: 0.4895 (KNN)
# precision: 0.6357 (SVC)
# precision: 0.7598 (Adaboost)
# precision: 0.7936 (GBM)
# precision: 0.7189 (XGBoost)
# precision: 0.7405 (LightGBM)

base_models(X, y, scoring="recall")
# recall: 0.0008 (LR)
# recall: 0.6643 (CART)
# recall: 0.6087 (RF)
# recall: 0.3677 (KNN)
# recall: 0.2581 (SVC)
# recall: 0.4805 (Adaboost)
# recall: 0.5657 (GBM)
# recall: 0.633 (XGBoost)
# recall: 0.6346 (LightGBM)

base_models(X, y, scoring="accuracy")
# accuracy: 0.6102 (LR)
# accuracy: 0.7323 (CART)
# accuracy: 0.7735 (RF)
# accuracy: 0.6041 (KNN)
# accuracy: 0.6517 (SVC)
# accuracy: 0.735 (Adaboost)
# accuracy: 0.7726 (GBM)
# accuracy: 0.7607 (XGBoost)
# accuracy: 0.7705 (LightGBM)

rf_params = {"max_depth": [list(range(1,11)), None],
                 "max_features": ["auto", "sqrt" "log2", 5, 7],
                 "min_samples_split": [15, 20],
                 "n_estimators": [10, 50, 100, 200, 300, 400]}

xgboost_params = {"learning_rate": [0.0001,0.001,0.01,0.1],
                      "max_depth": list(range(11)),
                      "n_estimators": [100,200,500,1000],
                      "colsample_bytree": [0.5, 1]}

lightgbm_params = {"learning_rate": [0.0001,0.001,0.01,0.1],
                       "n_estimators": [100, 200, 300, 500, 1000],
                       "colsample_bytree": [0.7, 1]}

classifiers = [ ("RF", RandomForestClassifier(), rf_params),
                ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
                ('LightGBM', LGBMClassifier(), lightgbm_params)]

def hyperparameter_optimization(X, y, cv=10, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X, y, cv=10, scoring="roc_auc")

# Hyperparameter Optimization....
# ########## RF ##########
# roc_auc (Before): 0.8466
# roc_auc (After): 0.8496
# RF best params: {'max_depth': None, 'max_features': 5, 'min_samples_split': 15, 'n_estimators': 400}
# ########## XGBoost ##########
# roc_auc (Before): 0.8445
# roc_auc (After): 0.8577
# XGBoost best params: {'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 100}
# ########## LightGBM ##########
# roc_auc (After): 0.8592
# LightGBM best params: {'colsample_bytree': 0.7, 'learning_rate': 0.01, 'n_estimators': 500}

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('RF', best_models["RF"]),
                                              ('XGBoost', best_models["XGBoost"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

voting_clf = voting_classifier(best_models, X, y)

# Accuracy: 0.7649572649572649
# F1Score: 0.6628948365241213
# ROC_AUC: 0.8389398788459821

# prediction
X.columns
random = X.sample(1, random_state=45)
voting_clf.predict(random)
joblib.dump(voting_clf, "voting_clf2.pkl") # the model is saved.

# new_model = joblib.load("voting_clf2.pkl") # the model is loaded.
# new_model.predict(random_user) # prediction is made.
