from julia import Distributed
Distributed.addprocs(20)
import pandas as pd
import interpretableai.iai as IAI
from pandas.api.types import CategoricalDtype
import numpy as np

data_path = "/pool001/htazi/Trauma/time_split_per_injury_new_morbidity/penetrating/"
train_X_time = pd.read_csv(data_path+"trauma_X_train_time_penetrating.csv")
test_X_time = pd.read_csv(data_path+"trauma_X_test_time_penetrating.csv")

names = ['age', 'gender', 'race1', 'signsoflife', 'sbp1', 'pulse1', 'oxysat1',
       'temp1', 'gcstot1', 'bleeding_disorder', 'current_chemotherapy',
       'congestive_heart_failure', 'current_smoker', 'chronic_renal_failure',
       'history_cva', 'diabetes', 'disseminated_cancer', 'copd', 'steroid',
       'cirrhosis', 'history_MI', 'history_pvd', 'hypertension_medication',
       'method_of_injury', "Head_severity", 'Face_severity', 'Neck_severity', 'Thorax_severity',
       'Abdomen_severity', 'Spine_severity', 'Upper_Extremity_severity',
       'Lower_Extremity_severity', 'Pelvis_Perineum_severity',
       'External_severity']
train_X_time = train_X_time.loc[:, names]
test_X_time = test_X_time.loc[:, names]
severity_categorical_var = [
    "Head_severity",
    "Face_severity",
    "Neck_severity",
    "Thorax_severity",
    "Abdomen_severity",
    "Spine_severity",
    "Upper_Extremity_severity",
    "Lower_Extremity_severity",
    "Pelvis_Perineum_severity",
]
# Dealing with Ordered Categorical Variables
#order_cat_severity = CategoricalDtype(categories=[0, 1, 2, 3, 4, 5, 6, 9], ordered=True)
#for col_name in severity_categorical_var:
#    train_X_time.loc[:, col_name] = train_X_time.loc[:, col_name].astype(
#        order_cat_severity
#    )
#    print("Transformed column '", col_name, "' to Ordered Categorical")

n_train = len(train_X_time)
X_time = pd.concat([train_X_time, test_X_time], axis=0)

k_neighbors_penetrating = 100
lnr_optimpute_penetrating = IAI.ImputationLearner(method="opt_knn", knn_k=k_neighbors_penetrating, cluster=True)
lnr_optimpute_penetrating.fit(X_time)