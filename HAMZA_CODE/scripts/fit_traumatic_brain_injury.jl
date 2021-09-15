using Random, CategoricalArrays, DataFrames, DataFramesMeta, StatsBase, CSV

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
X_tbi_penetrating = CSV.read("./X_tbi_penetrating.csv")[!, lnr_morbidity_penetrating.prb_.data.features.feature_names]
X_tbi_blunt = CSV.read("./X_tbi_blunt.csv")[!, lnr_morbidity_blunt.prb_.data.features.feature_names]
categorical!(X_tbi_penetrating)
categorical!(X_tbi_blunt)
y_pred_proba_tbi_morbid_penetrating = IAI.predict_proba(lnr_morbidity_penetrating, X_tbi_penetrating)
y_pred_proba_tbi_mortal_penetrating = IAI.predict_proba(lnr_mortality_penetrating, X_tbi_penetrating)
y_pred_proba_tbi_morbid_blunt = IAI.predict_proba(lnr_morbidity_blunt, X_tbi_blunt)
y_pred_proba_tbi_mortal_blunt = IAI.predict_proba(lnr_mortality_blunt, X_tbi_blunt)
CSV.write("./19_08_2020_tbi_outputs/y_pred_proba_tbi_morbid_penetrating.csv", y_pred_proba_tbi_morbid_penetrating)
CSV.write("./19_08_2020_tbi_outputs/y_pred_proba_tbi_mortal_penetrating.csv", y_pred_proba_tbi_mortal_penetrating)
CSV.write("./19_08_2020_tbi_outputs/y_pred_proba_tbi_morbid_blunt.csv", y_pred_proba_tbi_morbid_blunt)
CSV.write("./19_08_2020_tbi_outputs/y_pred_proba_tbi_mortal_blunt.csv", y_pred_proba_tbi_mortal_blunt)


### COMORBIDITIES ###
lnrs_comorb_path = "./outputs_comorb_2020_03_02/"
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
CSV.write("./19_08_2020_tbi_outputs/y_pred_proba_tbi_morbid_4_blunt.csv", y_pred_proba_tbi_morbid_4_blunt)
y_pred_proba_tbi_morbid_5_blunt = IAI.predict_proba(lnr_morbidity_5_blunt, X_tbi_blunt)
CSV.write("./19_08_2020_tbi_outputs/y_pred_proba_tbi_morbid_5_blunt.csv", y_pred_proba_tbi_morbid_5_blunt)
y_pred_proba_tbi_morbid_8_blunt = IAI.predict_proba(lnr_morbidity_8_blunt, X_tbi_blunt)
CSV.write("./19_08_2020_tbi_outputs/y_pred_proba_tbi_morbid_8_blunt.csv", y_pred_proba_tbi_morbid_8_blunt)
y_pred_proba_tbi_morbid_12_blunt = IAI.predict_proba(lnr_morbidity_12_blunt, X_tbi_blunt)
CSV.write("./19_08_2020_tbi_outputs/y_pred_proba_tbi_morbid_12_blunt.csv", y_pred_proba_tbi_morbid_12_blunt)
y_pred_proba_tbi_morbid_14_blunt = IAI.predict_proba(lnr_morbidity_14_blunt, X_tbi_blunt)
CSV.write("./19_08_2020_tbi_outputs/y_pred_proba_tbi_morbid_14_blunt.csv", y_pred_proba_tbi_morbid_14_blunt)
y_pred_proba_tbi_morbid_19_blunt = IAI.predict_proba(lnr_morbidity_19_blunt, X_tbi_blunt)
CSV.write("./19_08_2020_tbi_outputs/y_pred_proba_tbi_morbid_19_blunt.csv", y_pred_proba_tbi_morbid_19_blunt)
y_pred_proba_tbi_morbid_21_blunt = IAI.predict_proba(lnr_morbidity_21_blunt, X_tbi_blunt)
CSV.write("./19_08_2020_tbi_outputs/y_pred_proba_tbi_morbid_21_blunt.csv", y_pred_proba_tbi_morbid_21_blunt)
y_pred_proba_tbi_morbid_25_blunt = IAI.predict_proba(lnr_morbidity_25_blunt, X_tbi_blunt)
CSV.write("./19_08_2020_tbi_outputs/y_pred_proba_tbi_morbid_25_blunt.csv", y_pred_proba_tbi_morbid_25_blunt)
y_pred_proba_tbi_morbid_32_blunt = IAI.predict_proba(lnr_morbidity_32_blunt, X_tbi_blunt)
CSV.write("./19_08_2020_tbi_outputs/y_pred_proba_tbi_morbid_32_blunt.csv", y_pred_proba_tbi_morbid_32_blunt)

## PREDICTIONS PENETRATING
y_pred_proba_tbi_morbid_4_penetrating = IAI.predict_proba(lnr_morbidity_4_penetrating, X_tbi_penetrating)
CSV.write("./19_08_2020_tbi_outputs/y_pred_proba_tbi_morbid_4_penetrating.csv", y_pred_proba_tbi_morbid_4_penetrating)
y_pred_proba_tbi_morbid_5_penetrating = IAI.predict_proba(lnr_morbidity_5_penetrating, X_tbi_penetrating)
CSV.write("./19_08_2020_tbi_outputs/y_pred_proba_tbi_morbid_5_penetrating.csv", y_pred_proba_tbi_morbid_5_penetrating)
y_pred_proba_tbi_morbid_8_penetrating = IAI.predict_proba(lnr_morbidity_8_penetrating, X_tbi_penetrating)
CSV.write("./19_08_2020_tbi_outputs/y_pred_proba_tbi_morbid_8_penetrating.csv", y_pred_proba_tbi_morbid_8_penetrating)
y_pred_proba_tbi_morbid_12_penetrating = IAI.predict_proba(lnr_morbidity_12_penetrating, X_tbi_penetrating)
CSV.write("./19_08_2020_tbi_outputs/y_pred_proba_tbi_morbid_12_penetrating.csv", y_pred_proba_tbi_morbid_12_penetrating)
y_pred_proba_tbi_morbid_14_penetrating = IAI.predict_proba(lnr_morbidity_14_penetrating, X_tbi_penetrating)
CSV.write("./19_08_2020_tbi_outputs/y_pred_proba_tbi_morbid_14_penetrating.csv", y_pred_proba_tbi_morbid_14_penetrating)
y_pred_proba_tbi_morbid_19_penetrating = IAI.predict_proba(lnr_morbidity_19_penetrating, X_tbi_penetrating)
CSV.write("./19_08_2020_tbi_outputs/y_pred_proba_tbi_morbid_19_penetrating.csv", y_pred_proba_tbi_morbid_19_penetrating)
y_pred_proba_tbi_morbid_21_penetrating = IAI.predict_proba(lnr_morbidity_21_penetrating, X_tbi_penetrating)
CSV.write("./19_08_2020_tbi_outputs/y_pred_proba_tbi_morbid_21_penetrating.csv", y_pred_proba_tbi_morbid_21_penetrating)
y_pred_proba_tbi_morbid_25_penetrating = IAI.predict_proba(lnr_morbidity_25_penetrating, X_tbi_penetrating)
CSV.write("./19_08_2020_tbi_outputs/y_pred_proba_tbi_morbid_25_penetrating.csv", y_pred_proba_tbi_morbid_25_penetrating)
y_pred_proba_tbi_morbid_32_penetrating = IAI.predict_proba(lnr_morbidity_32_penetrating, X_tbi_penetrating)
CSV.write("./19_08_2020_tbi_outputs/y_pred_proba_tbi_morbid_32_penetrating.csv", y_pred_proba_tbi_morbid_32_penetrating)