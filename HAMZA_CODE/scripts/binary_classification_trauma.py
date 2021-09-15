# Author: Hamza Tazi Bouardi
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from binary_cart_toolkit import cart_toolkit
from binary_logistic_regression_toolkit import logistic_regression_toolkit
from binary_random_forest_toolkit import random_forest_toolkit
from binary_xgboost_toolkit import xgboost_toolkit
from binary_svm_toolkit import SVM_Toolkit


# We assume that the dataframe we read here is pre-processed,
# with the last column being the labels
data_path = "/pool001/htazi/Trauma/imputed_random_split_per_injury_without_severity_6/blunt/"
X_train_morbidity = pd.read_csv(data_path+"morbidity/train_X_morbid.csv")
y_train_morbidity= pd.read_csv(data_path+"morbidity/train_y_morbid.csv").iloc[:, 1]
X_test_morbidity = pd.read_csv(data_path+"morbidity/test_X_morbid.csv")
y_test_morbidity = pd.read_csv(data_path+"morbidity/test_y_morbid.csv").iloc[:, 1]
X_train_mortality = pd.read_csv(data_path+"mortality/train_X_mortal.csv")
y_train_mortality= pd.read_csv(data_path+"mortality/train_y_mortal.csv").iloc[:, 1]
X_test_mortality = pd.read_csv(data_path+"mortality/test_X_mortal.csv")
y_test_mortality = pd.read_csv(data_path+"mortality/test_y_mortal.csv").iloc[:, 1]
assert len(y_train_morbidity.unique()) == 2, \
    f"This dataset is not fit for binary classification as it has {len(y_train_morbidity.unique())} classes"

names = [
    "age", "gender", "race1", "signsoflife", "sbp1", "pulse1",
    "oxysat1", "temp1", "gcstot1", "bleeding_disorder", "current_chemotherapy",
    "congestive_heart_failure", "current_smoker", "chronic_renal_failure",
    "history_cva", "diabetes", "disseminated_cancer",  "copd", "steroid",
    "cirrhosis", "history_MI", "history_pvd", "hypertension_medication",
    "method_of_injury", "Head_severity", "Face_severity", "Neck_severity", "Thorax_severity", "Abdomen_severity",
    "Spine_severity", "Upper_Extremity_severity", "Lower_Extremity_severity",
    "Pelvis_Perineum_severity", "External_severity",
    #"severity_max", #"hemorrhage_ctrl_type", #"alcohol", #"tmode1", "acslevel"
]
X_train_morbidity = X_train_morbidity[names]
X_train_morbidity = pd.get_dummies(X_train_morbidity)
X_test_morbidity = X_test_morbidity[names]
X_test_morbidity = pd.get_dummies(X_test_morbidity)
X_train_mortality = X_train_mortality[names]
X_train_mortality = pd.get_dummies(X_train_mortality)
X_test_mortality = X_test_mortality[names]
X_test_mortality = pd.get_dummies(X_test_mortality)

columns_mortality_train = set(list(X_train_mortality.columns))
columns_morbidity_train = set(list(X_train_morbidity.columns))
columns_mortality_test = set(list(X_test_mortality.columns))
columns_morbidity_test = set(list(X_test_morbidity.columns))
columns_to_add_mortality_test = list(columns_mortality_train - columns_mortality_test)
columns_to_add_morbidity_test = list(columns_morbidity_train - columns_morbidity_test)
columns_to_add_mortality_train = list(columns_mortality_test - columns_mortality_train)
columns_to_add_morbidity_train = list(columns_morbidity_test - columns_morbidity_train)

if len(columns_to_add_mortality_test) != 0:
    for col_to_add in columns_to_add_mortality_test:
        X_test_mortality[col_to_add] = 0
else:
    pass

if len(columns_to_add_morbidity_test) != 0:
    for col_to_add in columns_to_add_morbidity_test:
        X_test_morbidity[col_to_add] = 0
else:
    pass

if len(columns_to_add_mortality_train) != 0:
    for col_to_add in columns_to_add_mortality_train:
        X_train_mortality[col_to_add] = 0
else:
    pass

if len(columns_to_add_morbidity_train) != 0:
    for col_to_add in columns_to_add_morbidity_train:
        X_train_morbidity[col_to_add] = 0
else:
    pass

assert set(list(X_train_mortality.columns)) == set(list(X_test_mortality.columns)), "Not the same columns in train/test"
columns_all = list(X_train_morbidity.columns)
X_train_morbidity = X_train_morbidity[columns_all]
X_train_mortality = X_train_mortality[columns_all]
X_test_morbidity = X_test_morbidity[columns_all]
X_test_mortality = X_test_mortality[columns_all]

dict_type_prediction = {0: "morbidity", 1: "mortality"}
for i, (X_train, y_train, X_test, y_test) in enumerate(
        [
            (X_train_morbidity, y_train_morbidity, X_test_morbidity, y_test_morbidity),
            (X_train_mortality, y_train_mortality, X_test_mortality, y_test_mortality)
        ]
):
    print(f"######################################## {dict_type_prediction[i]} ########################################")
    #### CART ####
    cart_model_best, roc_auc_cart, fpr_cart, tpr_cart = cart_toolkit(
            X_train, y_train, X_test, y_test
    )

    #### Logistic Regression ####
    logreg_model_best, roc_auc_logreg, fpr_logreg, tpr_logreg = logistic_regression_toolkit(
            X_train, y_train, X_test, y_test
    )

    #### Random Forest ####
    rf_model_best, roc_auc_rf, fpr_rf, tpr_rf = random_forest_toolkit(
            X_train, y_train, X_test, y_test
    )

    #### XGBoost ####
    xgb_model_best, roc_auc_xgb, fpr_xgb, tpr_xgb = xgboost_toolkit(
            X_train, y_train, X_test, y_test
    )

    #### Kernel SVM ####
    #SVM_toolkit_obj = SVM_Toolkit(X_train, X_test, y_train, y_test)
    #SVM_toolkit_obj.svm_toolkit()
    #svm_linear_model_best, roc_auc_svm_linear, fpr_svm_linear, tpr_svm_linear = (
    #        SVM_toolkit_obj.svm_linear_model_best,
    #        SVM_toolkit_obj.roc_auc_svm_linear,
    #        SVM_toolkit_obj.fpr_svm_linear,
    #        SVM_toolkit_obj.tpr_svm_linear
    #)
    #svm_rbf_model_best, roc_auc_svm_rbf, fpr_svm_rbf, tpr_svm_rbf = (
    #    SVM_toolkit_obj.svm_rbf_model_best,
    #    SVM_toolkit_obj.roc_auc_svm_rbf,
    #    SVM_toolkit_obj.fpr_svm_rbf,
    #    SVM_toolkit_obj.tpr_svm_rbf
    #)
    print("-----------------------------------------------------------------")