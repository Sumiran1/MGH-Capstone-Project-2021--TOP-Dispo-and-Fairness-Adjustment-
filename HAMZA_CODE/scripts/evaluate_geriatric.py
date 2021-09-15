import pandas as pd
from sklearn.metrics import roc_auc_score


# Transforming raw Geriatric data
df_trauma = pd.read_csv("/pool001/htazi/Trauma/GeriatricData/Geriatric_TOP2017TOHAMZAclean03152020.csv")
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
labels_penetrating = df_trauma_penetrating.loc[:, columns_to_drop]
labels_penetrating.to_csv("./labels_penetrating.csv")
labels_blunt = df_trauma_blunt.loc[:, columns_to_drop]
labels_blunt.to_csv("./labels_blunt.csv")
df_trauma_penetrating.drop(columns_to_drop, axis=1, inplace=True)
df_trauma_blunt.drop(columns_to_drop, axis=1, inplace=True)

columns_right_order = ['age', 'gender', 'race1', 'signsoflife', 'sbp1', 'pulse1', 'oxysat1',
       'temp1', 'gcstot1', 'bleeding_disorder', 'current_chemotherapy',
       'congestive_heart_failure', 'current_smoker', 'chronic_renal_failure',
       'history_cva', 'diabetes', 'disseminated_cancer', 'copd', 'steroid',
       'cirrhosis', 'history_MI', 'history_pvd', 'hypertension_medication',
       'method_of_injury', "Head_severity", 'Face_severity', 'Neck_severity', 'Thorax_severity',
       'Abdomen_severity', 'Spine_severity', 'Upper_Extremity_severity',
       'Lower_Extremity_severity', 'Pelvis_Perineum_severity',
       'External_severity']
assert len(df_trauma_blunt.columns) == len(columns_right_order)
assert len(df_trauma_penetrating.columns) == len(columns_right_order)
df_trauma_blunt = df_trauma_blunt.loc[:, columns_right_order]
df_trauma_penetrating = df_trauma_penetrating.loc[:, columns_right_order]
X_geriatric_penetrating = lnr_optimpute_penetrating.transform(df_trauma_penetrating)
X_geriatric_blunt = lnr_optimpute_blunt.transform(df_trauma_blunt)


mapping_gender_rev = {v:k for k,v in mapping_gender.items()}
mapping_race1_rev = {v:k for k,v in mapping_race1.items()}
mapping_signsoflife_rev = {v:k for k,v in mapping_signsoflife.items()}
mapping_method_of_injury_penetrating_rev = {v:k for k,v in mapping_method_of_injury_penetrating.items()}
mapping_method_of_injury_blunt_rev = {v:k for k,v in mapping_method_of_injury_blunt.items()}


##### Evaluating #####
# Results Comorbidities
for comorb in [4, 5, 8, 12, 14, 19, 21, 25, 32]:
    vars()[f"y_pred_proba_geriatric_morbid_{comorb}_blunt"] = pd.read_csv(
        f"./y_pred_proba_geriatric_morbid_{comorb}_blunt.csv"
    )
    vars()[f"y_pred_proba_geriatric_morbid_{comorb}_penetrating"] = pd.read_csv(
        f"./y_pred_proba_geriatric_morbid_{comorb}_penetrating.csv"
    )

for comorb in [4,5,8,12,14,19,21, 25, 32]:
    print(f"Penetrating - Comorbidity {comorb}: ",
        round(roc_auc_score(
            labels_penetrating[f"comorb_{comorb}"],
            vars()[f"y_pred_proba_geriatric_morbid_{comorb}_penetrating"].iloc[:, -1]
        ), 3))
    print(f"Blunt - Comorbidity {comorb}: ",
          round(roc_auc_score(
              labels_blunt[f"comorb_{comorb}"],
              vars()[f"y_pred_proba_geriatric_morbid_{comorb}_blunt"].iloc[:, -1]
          ), 3))
    print("------------------------------------------------------------------------------")

for comorb in [19,21, 25, 32]:
    for k in [1,2,3]:
        try:
            print(f"Penetrating - Comorbidity {comorb}, Group {k}: ",
            round(roc_auc_score(
                labels_penetrating[f"comorb_{comorb}"].iloc[vars()[f"idx_group_{k}_penetrating"]],
                vars()[f"y_pred_proba_geriatric_morbid_{comorb}_penetrating"].iloc[vars()[f"idx_group_{k}_penetrating"], -1]
            ), 3))
        except:
            print(f"Penetrating - Comorbidity {comorb}, Group {k}: N/A")
        try:
            print(f"Blunt - Comorbidity {comorb}, Group {k}: ",
                  round(roc_auc_score(
                      labels_blunt[f"comorb_{comorb}"].iloc[vars()[f"idx_group_{k}_blunt"]],
                      vars()[f"y_pred_proba_geriatric_morbid_{comorb}_blunt"].iloc[
                          vars()[f"idx_group_{k}_blunt"], -1]
                  ), 3))
        except:
            print(f"Blunt - Comorbidity {comorb}, Group {k}: N/A")
    print("------------------------------------------------------------------------------")


##### JULIA CODE
##########################################################################################
##########################################################################################
##########################################################################################
lnr_morbidity_blunt = IAI.read_json("./02_03_2020_outputs_random_split/seed=1___outcome=hosp_morbidity___minbucket=100___injury=blunt_auc754.json")
lnr_mortality_blunt = IAI.read_json("./02_03_2020_outputs_random_split/seed=1___outcome=hosp_mortality___minbucket=100___injury=blunt_auc890.json")
lnr_mortality_penetrating = IAI.read_json("./02_03_2020_outputs_random_split/seed=1___outcome=hosp_mortality___minbucket=100___injury=penetrating_auc941.json")
lnr_morbidity_penetrating = IAI.read_json("./02_03_2020_outputs_random_split/seed=1___outcome=hosp_morbidity___minbucket=100___injury=penetrating_auc777.json")
severity_categorical_var = [
           :Head_severity,
           :Face_severity,
           :Neck_severity,
           :Thorax_severity,
           :Abdomen_severity,
           :Spine_severity,
           :Upper_Extremity_severity,
           :Lower_Extremity_severity,
           :Pelvis_Perineum_severity,
           #:severity_max
       ]
X_geriatric_penetrating = CSV.read("./X_geriatric_penetrating.csv")[!, 2:35]
X_geriatric_blunt = CSV.read("./X_geriatric_blunt.csv")[!, 2:35]
#for col_name in severity_categorical_var
#   X_geriatric_penetrating[!, col_name] = CategoricalArray(X_geriatric_penetrating[!, col_name], ordered=true)
#   X_geriatric_blunt[!, col_name] = CategoricalArray(X_geriatric_blunt[!, col_name], ordered=true)
#end
#categorical!(X_geriatric_penetrating)
#categorical!(X_geriatric_blunt)
#CSV.write("./X_geriatric_penetrating.csv", X_geriatric_penetrating)
#CSV.write("./X_geriatric_blunt.csv", X_geriatric_blunt)
y_pred_proba_geriatric_morbid_penetrating = IAI.predict_proba(lnr_morbidity_penetrating, X_geriatric_penetrating)
y_pred_proba_geriatric_mortal_penetrating = IAI.predict_proba(lnr_mortality_penetrating, X_geriatric_penetrating)
y_pred_proba_geriatric_morbid_blunt = IAI.predict_proba(lnr_morbidity_blunt, X_geriatric_blunt)
y_pred_proba_geriatric_mortal_blunt = IAI.predict_proba(lnr_mortality_blunt, X_geriatric_blunt)
CSV.write("./y_pred_proba_geriatric_morbid_penetrating.csv", y_pred_proba_geriatric_morbid_penetrating)
CSV.write("./y_pred_proba_geriatric_mortal_penetrating.csv", y_pred_proba_geriatric_mortal_penetrating)
CSV.write("./y_pred_proba_geriatric_morbid_blunt.csv", y_pred_proba_geriatric_morbid_blunt)
CSV.write("./y_pred_proba_geriatric_mortal_blunt.csv", y_pred_proba_geriatric_mortal_blunt)


### COMORBIDITIES ###
lnrs_comorb_path = "../../comorbidities/oct_scripts/outputs_comorb_2020_03_02/"
lnr_morbidity_4_blunt = IAI.read_json(lnrs_comorb_path*"seed=1___outcome=hosp_morbidity_4___minbucket=100___injury=blunt_auc752.json")
lnr_morbidity_5_blunt = IAI.read_json(lnrs_comorb_path*"seed=1___outcome=hosp_morbidity_5___minbucket=100___injury=blunt_auc787.json")
lnr_morbidity_8_blunt = IAI.read_json(lnrs_comorb_path*"seed=1___outcome=hosp_morbidity_8___minbucket=100___injury=blunt_auc831.json")
lnr_morbidity_12_blunt = IAI.read_json(lnrs_comorb_path*"seed=1___outcome=hosp_morbidity_12___minbucket=100___injury=blunt_auc736.json")
lnr_morbidity_14_blunt = IAI.read_json(lnrs_comorb_path*"seed=1___outcome=hosp_morbidity_14___minbucket=100___injury=blunt_auc737.json")
lnr_morbidity_19_blunt = IAI.read_json(lnrs_comorb_path*"seed=1___outcome=hosp_morbidity_19___minbucket=100___injury=blunt_auc750.json")
lnr_morbidity_21_blunt = IAI.read_json(lnrs_comorb_path*"seed=1___outcome=hosp_morbidity_21___minbucket=100___injury=blunt_auc689.json")
lnr_morbidity_25_blunt = IAI.read_json(lnrs_comorb_path*"seed=1___outcome=hosp_morbidity_25___minbucket=100___injury=blunt_auc739.json")
lnr_morbidity_32_blunt = IAI.read_json(lnrs_comorb_path*"seed=1___outcome=hosp_morbidity_32___minbucket=100___injury=blunt_auc751.json")


lnr_morbidity_4_penetrating = IAI.read_json(lnrs_comorb_path*"seed=1___outcome=hosp_morbidity_4___minbucket=100___injury=penetrating_auc784.json")
lnr_morbidity_5_penetrating = IAI.read_json(lnrs_comorb_path*"seed=1___outcome=hosp_morbidity_5___minbucket=100___injury=penetrating_auc756.json")
lnr_morbidity_8_penetrating = IAI.read_json(lnrs_comorb_path*"seed=1___outcome=hosp_morbidity_8___minbucket=100___injury=penetrating_auc835.json")
lnr_morbidity_12_penetrating = IAI.read_json(lnrs_comorb_path*"seed=1___outcome=hosp_morbidity_12___minbucket=100___injury=penetrating_auc812.json")
lnr_morbidity_14_penetrating = IAI.read_json(lnrs_comorb_path*"seed=1___outcome=hosp_morbidity_14___minbucket=100___injury=penetrating_auc741.json")
lnr_morbidity_19_penetrating = IAI.read_json(lnrs_comorb_path*"seed=1___outcome=hosp_morbidity_19___minbucket=100___injury=penetrating_auc801.json")
lnr_morbidity_21_penetrating = IAI.read_json(lnrs_comorb_path*"seed=1___outcome=hosp_morbidity_21___minbucket=100___injury=penetrating_auc722.json")
lnr_morbidity_25_penetrating = IAI.read_json(lnrs_comorb_path*"seed=1___outcome=hosp_morbidity_25___minbucket=100___injury=penetrating_auc735.json")
lnr_morbidity_32_penetrating = IAI.read_json(lnrs_comorb_path*"seed=1___outcome=hosp_morbidity_32___minbucket=100___injury=penetrating_auc817.json")

## PREDICTIONS BLUNT
y_pred_proba_geriatric_morbid_4_blunt = IAI.predict_proba(lnr_morbidity_4_blunt, X_geriatric_blunt)
CSV.write("./y_pred_proba_geriatric_morbid_4_blunt.csv", y_pred_proba_geriatric_morbid_4_blunt)
y_pred_proba_geriatric_morbid_5_blunt = IAI.predict_proba(lnr_morbidity_5_blunt, X_geriatric_blunt)
CSV.write("./y_pred_proba_geriatric_morbid_5_blunt.csv", y_pred_proba_geriatric_morbid_5_blunt)
y_pred_proba_geriatric_morbid_8_blunt = IAI.predict_proba(lnr_morbidity_8_blunt, X_geriatric_blunt)
CSV.write("./y_pred_proba_geriatric_morbid_8_blunt.csv", y_pred_proba_geriatric_morbid_8_blunt)
y_pred_proba_geriatric_morbid_12_blunt = IAI.predict_proba(lnr_morbidity_12_blunt, X_geriatric_blunt)
CSV.write("./y_pred_proba_geriatric_morbid_12_blunt.csv", y_pred_proba_geriatric_morbid_12_blunt)
y_pred_proba_geriatric_morbid_14_blunt = IAI.predict_proba(lnr_morbidity_14_blunt, X_geriatric_blunt)
CSV.write("./y_pred_proba_geriatric_morbid_14_blunt.csv", y_pred_proba_geriatric_morbid_14_blunt)
y_pred_proba_geriatric_morbid_19_blunt = IAI.predict_proba(lnr_morbidity_19_blunt, X_geriatric_blunt)
CSV.write("./y_pred_proba_geriatric_morbid_19_blunt.csv", y_pred_proba_geriatric_morbid_19_blunt)
y_pred_proba_geriatric_morbid_21_blunt = IAI.predict_proba(lnr_morbidity_21_blunt, X_geriatric_blunt)
CSV.write("./y_pred_proba_geriatric_morbid_21_blunt.csv", y_pred_proba_geriatric_morbid_21_blunt)
y_pred_proba_geriatric_morbid_25_blunt = IAI.predict_proba(lnr_morbidity_25_blunt, X_geriatric_blunt)
CSV.write("./y_pred_proba_geriatric_morbid_25_blunt.csv", y_pred_proba_geriatric_morbid_25_blunt)
y_pred_proba_geriatric_morbid_32_blunt = IAI.predict_proba(lnr_morbidity_32_blunt, X_geriatric_blunt)
CSV.write("./y_pred_proba_geriatric_morbid_32_blunt.csv", y_pred_proba_geriatric_morbid_32_blunt)

## PREDICTIONS PENETRATING
y_pred_proba_geriatric_morbid_4_penetrating = IAI.predict_proba(lnr_morbidity_4_penetrating, X_geriatric_penetrating)
CSV.write("./y_pred_proba_geriatric_morbid_4_penetrating.csv", y_pred_proba_geriatric_morbid_4_penetrating)
y_pred_proba_geriatric_morbid_5_penetrating = IAI.predict_proba(lnr_morbidity_5_penetrating, X_geriatric_penetrating)
CSV.write("./y_pred_proba_geriatric_morbid_5_penetrating.csv", y_pred_proba_geriatric_morbid_5_penetrating)
y_pred_proba_geriatric_morbid_8_penetrating = IAI.predict_proba(lnr_morbidity_8_penetrating, X_geriatric_penetrating)
CSV.write("./y_pred_proba_geriatric_morbid_8_penetrating.csv", y_pred_proba_geriatric_morbid_8_penetrating)
y_pred_proba_geriatric_morbid_12_penetrating = IAI.predict_proba(lnr_morbidity_12_penetrating, X_geriatric_penetrating)
CSV.write("./y_pred_proba_geriatric_morbid_12_penetrating.csv", y_pred_proba_geriatric_morbid_12_penetrating)
y_pred_proba_geriatric_morbid_14_penetrating = IAI.predict_proba(lnr_morbidity_14_penetrating, X_geriatric_penetrating)
CSV.write("./y_pred_proba_geriatric_morbid_14_penetrating.csv", y_pred_proba_geriatric_morbid_14_penetrating)
y_pred_proba_geriatric_morbid_19_penetrating = IAI.predict_proba(lnr_morbidity_19_penetrating, X_geriatric_penetrating)
CSV.write("./y_pred_proba_geriatric_morbid_19_penetrating.csv", y_pred_proba_geriatric_morbid_19_penetrating)
y_pred_proba_geriatric_morbid_21_penetrating = IAI.predict_proba(lnr_morbidity_21_penetrating, X_geriatric_penetrating)
CSV.write("./y_pred_proba_geriatric_morbid_21_penetrating.csv", y_pred_proba_geriatric_morbid_21_penetrating)
y_pred_proba_geriatric_morbid_25_penetrating = IAI.predict_proba(lnr_morbidity_25_penetrating, X_geriatric_penetrating)
CSV.write("./y_pred_proba_geriatric_morbid_25_penetrating.csv", y_pred_proba_geriatric_morbid_25_penetrating)
y_pred_proba_geriatric_morbid_32_penetrating = IAI.predict_proba(lnr_morbidity_32_penetrating, X_geriatric_penetrating)
CSV.write("./y_pred_proba_geriatric_morbid_32_penetrating.csv", y_pred_proba_geriatric_morbid_32_penetrating)