import pandas as pd
from sklearn.metrics import roc_auc_score
from julia import Distributed
Distributed.addprocs(14)
import interpretableai.iai as IAI
from pandas.api.types import CategoricalDtype


# Transforming raw TBI data
df_trauma = pd.read_csv("/pool001/htazi/Trauma/TBI_Data/TBI_TOP2017TOHAMZAclean_03112020.csv")
mapping_gender = {"Female": 1, "Male": 0}
mapping_race1 = {'Other Race': 0, 'Black or African American': 1, 'White': 2, 'American Indian': 3,
               'Native Hawaiian or Other Pacific Islander': 4, 'Asian': 5}
mapping_signsoflife = {'Unknown':-1, 'Arrived with signs of life': 1, 'Arrived with NO signs of life': 0}
df_trauma["gender"] = df_trauma["gender"].map(mapping_gender)
df_trauma["race1"] = df_trauma["race1"].map(mapping_race1)
df_trauma["signsoflife"] = df_trauma["signsoflife"].map(mapping_signsoflife)

# Creating blunt and penetrating
mapping_method_of_injury_penetrating = {'Penetrating - Stab Wound': 1, 'Penetrating - Gunshot Wound': 2,
                                       'Penetrating - Other/Mixed':3}
mapping_method_of_injury_blunt = {'Blunt - MVT occupant': 1, 'Blunt - Fall': 2, 'Blunt - MVT motorcyclist': 3,
 'Blunt - Other': 4, 'Blunt - MVT Pedal cyclist/pedestrian': 5}
df_trauma_blunt = df_trauma[df_trauma.method_of_injury.str.contains("Blunt")]
df_trauma_penetrating = df_trauma[df_trauma.method_of_injury.str.contains("Penetrating")]
df_trauma_penetrating["method_of_injury"] = df_trauma_penetrating.method_of_injury.map(mapping_method_of_injury_penetrating)
df_trauma_blunt["method_of_injury"] = df_trauma_blunt.method_of_injury.map(mapping_method_of_injury_blunt)
columns_right_order = ['age', 'gender', 'race1', 'signsoflife', 'sbp1', 'pulse1', 'oxysat1',
       'temp1', 'gcstot1', 'bleeding_disorder', 'current_chemotherapy',
       'congestive_heart_failure', 'current_smoker', 'chronic_renal_failure',
       'history_cva', 'diabetes', 'disseminated_cancer', 'copd', 'steroid',
       'cirrhosis', 'history_MI', 'history_pvd', 'hypertension_medication',
       'method_of_injury', "Head_severity", 'Face_severity', 'Neck_severity', 'Thorax_severity',
       'Abdomen_severity', 'Spine_severity', 'Upper_Extremity_severity',
       'Lower_Extremity_severity', 'Pelvis_Perineum_severity',
       'External_severity']

labels = [
    'unplannedintubation', 'cpr', 'hospitalmortality', 'dvt', 'pulmemb', 'totalmorbidity', 'severesepsis', 
    'ards', 'tbigcsseverity', 'organspacesurgicalsiteinfection', 'deepssi', 'aki'
]
labels_tbi_renaming = {
    "unplannedintubation": "comorb_25", "cpr": "comorb_8", "dvt": "comorb_14", "pulmemb": "comorb_21",
    "severesepsis": "comorb_32", "ards": "comorb_5", "tbigcsseverity": "comorb_XX", "deepssi": "comorb_12",
    "aki": "comorb_4", "organspacesurgicalsiteinfection": "comorb_19",
}
labels_tbi_penetrating = df_trauma_penetrating.loc[:, labels]
new_columns_penetrating = [
    labels_tbi_renaming[x]
    if x in labels_tbi_renaming.keys() else x
    for x in labels_tbi_penetrating.columns
]

labels_tbi_penetrating.columns = new_columns_penetrating
labels_tbi_penetrating.to_csv("./labels_tbi_penetrating.csv")

labels_tbi_blunt = df_trauma_blunt.loc[:, labels]
new_columns_blunt = [
    labels_tbi_renaming[x]
    if x in labels_tbi_renaming.keys() else x
    for x in labels_tbi_blunt.columns
]

labels_tbi_blunt.columns = new_columns_blunt
labels_tbi_blunt.to_csv("./labels_tbi_blunt.csv")

df_trauma_penetrating.drop(labels, axis=1, inplace=True)
df_trauma_blunt.drop(labels, axis=1, inplace=True)
columns_not_in_data_penetrating = set(columns_right_order) - set(df_trauma_penetrating.columns)
for col in columns_not_in_data_penetrating:
    print(col)
    df_trauma_penetrating[col] = 0

columns_not_in_data_blunt = set(columns_right_order) - set(df_trauma_blunt.columns)
for col in columns_not_in_data_blunt:
    print(col)
    df_trauma_blunt[col] = 0

df_trauma_blunt = df_trauma_blunt.loc[:, columns_right_order]
df_trauma_penetrating = df_trauma_penetrating.loc[:, columns_right_order]
assert len(df_trauma_blunt.columns) == len(columns_right_order)
assert len(df_trauma_penetrating.columns) == len(columns_right_order)
k_neighbors_blunt = 100
lnr_optimpute_blunt = IAI.ImputationLearner(method="opt_knn", knn_k=k_neighbors_blunt, cluster=True)
#lnr_optimpute_blunt.fit(df_trauma_blunt)
X_tbi_blunt = lnr_optimpute_blunt.fit_transform(df_trauma_blunt)

k_neighbors_penetrating = 100
lnr_optimpute_penetrating = IAI.ImputationLearner(method="opt_knn", knn_k=k_neighbors_penetrating, cluster=True)
lnr_optimpute_penetrating.fit(df_trauma_penetrating)
X_tbi_penetrating = lnr_optimpute_penetrating.transform(df_trauma_penetrating)

mapping_gender_rev = {v: k for k, v in mapping_gender.items()}
mapping_race1_rev = {v: k for k, v in mapping_race1.items()}
mapping_signsoflife_rev = {v: k for k, v in mapping_signsoflife.items()}
mapping_method_of_injury_penetrating_rev = {v: k for k, v in mapping_method_of_injury_penetrating.items()}
mapping_method_of_injury_blunt_rev = {v: k for k, v in mapping_method_of_injury_blunt.items()}

X_tbi_penetrating["gender"] = X_tbi_penetrating.gender.map(mapping_gender_rev)
X_tbi_penetrating["race1"] = X_tbi_penetrating.race1.map(mapping_race1_rev)
X_tbi_penetrating["signsoflife"] = X_tbi_penetrating.signsoflife.map(mapping_signsoflife_rev)
X_tbi_penetrating["method_of_injury"] = X_tbi_penetrating.method_of_injury.map(mapping_method_of_injury_penetrating_rev)

X_tbi_blunt["gender"] = X_tbi_blunt.gender.map(mapping_gender_rev)
X_tbi_blunt["race1"] = X_tbi_blunt.race1.map(mapping_race1_rev)
X_tbi_blunt["signsoflife"] = X_tbi_blunt.signsoflife.map(mapping_signsoflife_rev)
X_tbi_blunt["method_of_injury"] = X_tbi_blunt.method_of_injury.map(mapping_method_of_injury_blunt_rev)

X_tbi_blunt.to_csv("./X_tbi_blunt.csv")
X_tbi_penetrating.to_csv("./X_tbi_penetrating.csv")