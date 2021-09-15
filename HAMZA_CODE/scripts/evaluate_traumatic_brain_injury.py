import pandas as pd
from sklearn.metrics import roc_auc_score
from julia import Distributed
Distributed.addprocs(20)
import interpretableai.iai as IAI
from pandas.api.types import CategoricalDtype

labels_penetrating = pd.read_csv("./labels_tbi_penetrating.csv")
labels_blunt = pd.read_csv("./labels_tbi_blunt.csv")


##### Evaluating #####
# Results Morbidity & Mortality
y_pred_proba_tbi_morbid_blunt = pd.read_csv("./19_08_2020_tbi_outputs/y_pred_proba_tbi_morbid_blunt.csv")
y_pred_proba_tbi_mortal_blunt = pd.read_csv("./19_08_2020_tbi_outputs/y_pred_proba_tbi_mortal_blunt.csv")
print(f"Blunt Morbidity:",
    round(roc_auc_score(
        labels_blunt[f"totalmorbidity"],
        y_pred_proba_tbi_morbid_blunt.iloc[:, -1]
    ), 3))
print(f"Blunt Mortality:",
      round(roc_auc_score(
          labels_blunt[f"hospitalmortality"],
          y_pred_proba_tbi_mortal_blunt.iloc[:, -1]
      ), 3))
print("------------------------------------------------------------------------------")
y_pred_proba_tbi_morbid_penetrating = pd.read_csv("./19_08_2020_tbi_outputs/y_pred_proba_tbi_morbid_penetrating.csv")
y_pred_proba_tbi_mortal_penetrating = pd.read_csv("./19_08_2020_tbi_outputs/y_pred_proba_tbi_mortal_penetrating.csv")
print(f"Penetrating Morbidity:",
    round(roc_auc_score(
        labels_penetrating[f"totalmorbidity"],
        y_pred_proba_tbi_morbid_penetrating.iloc[:, -1]
    ), 3))
print(f"Penetrating Mortality:",
      round(roc_auc_score(
          labels_penetrating[f"hospitalmortality"],
          y_pred_proba_tbi_mortal_penetrating.iloc[:, -1]
      ), 3))
print("------------------------------------------------------------------------------")



# Results Comorbidities
for comorb in [4, 5, 8, 12, 14, 19, 21, 25, 32]:
    vars()[f"y_pred_proba_tbi_morbid_{comorb}_blunt"] = pd.read_csv(
        f"./19_08_2020_tbi_outputs/y_pred_proba_tbi_morbid_{comorb}_blunt.csv"
    )
    vars()[f"y_pred_proba_tbi_morbid_{comorb}_penetrating"] = pd.read_csv(
        f"./19_08_2020_tbi_outputs/y_pred_proba_tbi_morbid_{comorb}_penetrating.csv"
    )

for comorb in [4, 5, 8, 12, 14, 19, 21, 25, 32]:
    print(f"Penetrating - Comorbidity {comorb}: ",
        round(roc_auc_score(
            labels_penetrating[f"comorb_{comorb}"],
            vars()[f"y_pred_proba_tbi_morbid_{comorb}_penetrating"].iloc[:, -1]
        ), 3))
    print(f"Blunt - Comorbidity {comorb}: ",
          round(roc_auc_score(
              labels_blunt[f"comorb_{comorb}"],
              vars()[f"y_pred_proba_tbi_morbid_{comorb}_blunt"].iloc[:, -1]
          ), 3))
    print("------------------------------------------------------------------------------")

# for comorb in [19, 21, 25, 32]:
#     for k in [1, 2, 3]:
#         try:
#             print(f"Penetrating - Comorbidity {comorb}, Group {k}: ",
#             round(roc_auc_score(
#                 labels_penetrating[f"comorb_{comorb}"].iloc[vars()[f"idx_group_{k}_penetrating"]],
#                 vars()[f"y_pred_proba_tbi_morbid_{comorb}_penetrating"].iloc[vars()[f"idx_group_{k}_penetrating"], -1]
#             ), 3))
#         except:
#             print(f"Penetrating - Comorbidity {comorb}, Group {k}: N/A")
#         try:
#             print(f"Blunt - Comorbidity {comorb}, Group {k}: ",
#                   round(roc_auc_score(
#                       labels_blunt[f"comorb_{comorb}"].iloc[vars()[f"idx_group_{k}_blunt"]],
#                       vars()[f"y_pred_proba_tbi_morbid_{comorb}_blunt"].iloc[
#                           vars()[f"idx_group_{k}_blunt"], -1]
#                   ), 3))
#         except:
#             print(f"Blunt - Comorbidity {comorb}, Group {k}: N/A")
#     print("------------------------------------------------------------------------------")


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
X_tbi_penetrating = CSV.read("./X_tbi_penetrating.csv")[!, 2:35]
X_tbi_blunt = CSV.read("./X_tbi_blunt.csv")[!, 2:35]
#for col_name in severity_categorical_var
#   X_tbi_penetrating[!, col_name] = CategoricalArray(X_tbi_penetrating[!, col_name], ordered=true)
#   X_tbi_blunt[!, col_name] = CategoricalArray(X_tbi_blunt[!, col_name], ordered=true)
#end
#categorical!(X_tbi_penetrating)
#categorical!(X_tbi_blunt)
#CSV.write("./X_tbi_penetrating.csv", X_tbi_penetrating)
#CSV.write("./X_tbi_blunt.csv", X_tbi_blunt)
y_pred_proba_tbi_morbid_penetrating = IAI.predict_proba(lnr_morbidity_penetrating, X_tbi_penetrating)
y_pred_proba_tbi_mortal_penetrating = IAI.predict_proba(lnr_mortality_penetrating, X_tbi_penetrating)
y_pred_proba_tbi_morbid_blunt = IAI.predict_proba(lnr_morbidity_blunt, X_tbi_blunt)
y_pred_proba_tbi_mortal_blunt = IAI.predict_proba(lnr_mortality_blunt, X_tbi_blunt)
CSV.write("./y_pred_proba_tbi_morbid_penetrating.csv", y_pred_proba_tbi_morbid_penetrating)
CSV.write("./y_pred_proba_tbi_mortal_penetrating.csv", y_pred_proba_tbi_mortal_penetrating)
CSV.write("./y_pred_proba_tbi_morbid_blunt.csv", y_pred_proba_tbi_morbid_blunt)
CSV.write("./y_pred_proba_tbi_mortal_blunt.csv", y_pred_proba_tbi_mortal_blunt)


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
y_pred_proba_tbi_morbid_4_blunt = IAI.predict_proba(lnr_morbidity_4_blunt, X_tbi_blunt)
CSV.write("./y_pred_proba_tbi_morbid_4_blunt.csv", y_pred_proba_tbi_morbid_4_blunt)
y_pred_proba_tbi_morbid_5_blunt = IAI.predict_proba(lnr_morbidity_5_blunt, X_tbi_blunt)
CSV.write("./y_pred_proba_tbi_morbid_5_blunt.csv", y_pred_proba_tbi_morbid_5_blunt)
y_pred_proba_tbi_morbid_8_blunt = IAI.predict_proba(lnr_morbidity_8_blunt, X_tbi_blunt)
CSV.write("./y_pred_proba_tbi_morbid_8_blunt.csv", y_pred_proba_tbi_morbid_8_blunt)
y_pred_proba_tbi_morbid_12_blunt = IAI.predict_proba(lnr_morbidity_12_blunt, X_tbi_blunt)
CSV.write("./y_pred_proba_tbi_morbid_12_blunt.csv", y_pred_proba_tbi_morbid_12_blunt)
y_pred_proba_tbi_morbid_14_blunt = IAI.predict_proba(lnr_morbidity_14_blunt, X_tbi_blunt)
CSV.write("./y_pred_proba_tbi_morbid_14_blunt.csv", y_pred_proba_tbi_morbid_14_blunt)
y_pred_proba_tbi_morbid_19_blunt = IAI.predict_proba(lnr_morbidity_19_blunt, X_tbi_blunt)
CSV.write("./y_pred_proba_tbi_morbid_19_blunt.csv", y_pred_proba_tbi_morbid_19_blunt)
y_pred_proba_tbi_morbid_21_blunt = IAI.predict_proba(lnr_morbidity_21_blunt, X_tbi_blunt)
CSV.write("./y_pred_proba_tbi_morbid_21_blunt.csv", y_pred_proba_tbi_morbid_21_blunt)
y_pred_proba_tbi_morbid_25_blunt = IAI.predict_proba(lnr_morbidity_25_blunt, X_tbi_blunt)
CSV.write("./y_pred_proba_tbi_morbid_25_blunt.csv", y_pred_proba_tbi_morbid_25_blunt)
y_pred_proba_tbi_morbid_32_blunt = IAI.predict_proba(lnr_morbidity_32_blunt, X_tbi_blunt)
CSV.write("./y_pred_proba_tbi_morbid_32_blunt.csv", y_pred_proba_tbi_morbid_32_blunt)

## PREDICTIONS PENETRATING
y_pred_proba_tbi_morbid_4_penetrating = IAI.predict_proba(lnr_morbidity_4_penetrating, X_tbi_penetrating)
CSV.write("./y_pred_proba_tbi_morbid_4_penetrating.csv", y_pred_proba_tbi_morbid_4_penetrating)
y_pred_proba_tbi_morbid_5_penetrating = IAI.predict_proba(lnr_morbidity_5_penetrating, X_tbi_penetrating)
CSV.write("./y_pred_proba_tbi_morbid_5_penetrating.csv", y_pred_proba_tbi_morbid_5_penetrating)
y_pred_proba_tbi_morbid_8_penetrating = IAI.predict_proba(lnr_morbidity_8_penetrating, X_tbi_penetrating)
CSV.write("./y_pred_proba_tbi_morbid_8_penetrating.csv", y_pred_proba_tbi_morbid_8_penetrating)
y_pred_proba_tbi_morbid_12_penetrating = IAI.predict_proba(lnr_morbidity_12_penetrating, X_tbi_penetrating)
CSV.write("./y_pred_proba_tbi_morbid_12_penetrating.csv", y_pred_proba_tbi_morbid_12_penetrating)
y_pred_proba_tbi_morbid_14_penetrating = IAI.predict_proba(lnr_morbidity_14_penetrating, X_tbi_penetrating)
CSV.write("./y_pred_proba_tbi_morbid_14_penetrating.csv", y_pred_proba_tbi_morbid_14_penetrating)
y_pred_proba_tbi_morbid_19_penetrating = IAI.predict_proba(lnr_morbidity_19_penetrating, X_tbi_penetrating)
CSV.write("./y_pred_proba_tbi_morbid_19_penetrating.csv", y_pred_proba_tbi_morbid_19_penetrating)
y_pred_proba_tbi_morbid_21_penetrating = IAI.predict_proba(lnr_morbidity_21_penetrating, X_tbi_penetrating)
CSV.write("./y_pred_proba_tbi_morbid_21_penetrating.csv", y_pred_proba_tbi_morbid_21_penetrating)
y_pred_proba_tbi_morbid_25_penetrating = IAI.predict_proba(lnr_morbidity_25_penetrating, X_tbi_penetrating)
CSV.write("./y_pred_proba_tbi_morbid_25_penetrating.csv", y_pred_proba_tbi_morbid_25_penetrating)
y_pred_proba_tbi_morbid_32_penetrating = IAI.predict_proba(lnr_morbidity_32_penetrating, X_tbi_penetrating)
CSV.write("./y_pred_proba_tbi_morbid_32_penetrating.csv", y_pred_proba_tbi_morbid_32_penetrating)