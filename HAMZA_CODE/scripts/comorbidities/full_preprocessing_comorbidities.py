#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

df_trauma = pd.read_csv("/pool001/htazi/Trauma/original_data/TQIP_2010_2016_Merged_MGHTrauma2019Jan.csv")
initial_length = len(df_trauma)
df_trauma.head()


## General preprocessing
allowed_eddisp = [
  "Operating Room",
  # "Transferred to another hospital",
  "Observation unit (unit that provides &lt; 24 hour stays)",
  "Intensive Care Unit (ICU)",
  "Telemetry/step-down unit (less acuity than ICU)",
  "Floor bed (general admission, non specialty unit bed)"
  # "Home without services",
  # "Other (jail, institutional care facility, mental health, etc)",
  # "Home with services",
  # "Left against medical advice",
]
df_trauma = df_trauma[df_trauma["eddisp"].isin(allowed_eddisp)]
df_trauma.loc[(df_trauma.tmode1.isnull()) & ~(df_trauma.tmode2.isnull()), "tmode1"] = df_trauma.loc[
    (df_trauma.tmode1.isnull()) & ~(df_trauma.tmode2.isnull()), "tmode2"
]


comorkeys = [x for x in df_trauma.columns if "comorkey" in x]
complkeys = [x for x in df_trauma.columns if "complkey" in x]
predotkeys = [x for x in df_trauma.columns if "predot" in x]
severitykeys = [x for x in df_trauma.columns if "severity" in x]
columns_kept_daisy = (
    comorkeys
    + complkeys
    + predotkeys
    + severitykeys
    + [
        "inc_key", "issais", "age", "gender", "race1", "ethnic", "acslevel",
        "tmode1", "tmode2", "transfer", "alcohol", "drug1", "signsoflife",
        "sbp1", "sbp2", "pulse1", "pulse2", "rr1", "rr2",
        "oxysat1", "oxysat2", "temp1", "gcstot1", "gcstot2",
        "ecode", "icd10_primary_ecode", "icd10_additonal_ecode",
        "eddisp", "hospdisp", "yoadmit", "teachsta", "region", "hemorrhage_ctrl_type"
    ]
)
df_trauma = df_trauma[columns_kept_daisy]


##### Mapping values to NaNs #####
for col_severity in severitykeys:
    df_trauma[col_severity] = df_trauma[col_severity].replace({9: np.nan})
df_trauma = df_trauma.replace({
    "Not Applicable BIU 1": np.nan,
    "Not Known/Not Recorded BIU 2": np.nan,
    "Not Applicable": np.nan,
    -99: np.nan,
    -1: np.nan,
    -2: np.nan,
})


## Comorbidities
for new_column, value in zip(
    ["alcohol_use_disorder", "bleeding_disorder", "current_chemotherapy", "congestive_heart_failure",
    "current_smoker", "chronic_renal_failure", "history_cva", "diabetes", "disseminated_cancer", "copd",
    "steroid", "cirrhosis", "drug_use_disorder", "history_MI", "history_pvd", "hypertension_medication"],
    [2, 4, 5, 7, 8, 9, 10, 11, 12, 23, 24, 25, 28, 17, 18, 19]
):
    df_trauma[new_column] = 0
    df_trauma.loc[
        ((df_trauma["comorkey1"] == value) | (df_trauma["comorkey2"] == value) |
        (df_trauma["comorkey3"] == value) | (df_trauma["comorkey4"] == value) |
        (df_trauma["comorkey5"] == value) | (df_trauma["comorkey6"] == value) |
        (df_trauma["comorkey7"] == value) | (df_trauma["comorkey8"] == value) |
        (df_trauma["comorkey9"] == value) | (df_trauma["comorkey10"] == value) |
        (df_trauma["comorkey11"] == value) | (df_trauma["comorkey12"] == value)),
        new_column
    ] = 1
    print(f"Created new column {new_column}")
df_trauma = df_trauma.loc[:, [col for col in df_trauma.columns if "comorkey" not in col]]


## Morbidities
# allowed_morbidities = [4, 5, 8, 11, 12, 14, 15, 18, 19, 21, 22, 23, 25, 30, 31, 32, 35]
# Modification of allowed morbidities on 15/12/2019: deleted 11, 15, 18, 22, 23, 30, 31,35
allowed_morbidities = [4, 5, 8, 12, 14, 19, 21, 25, 32]
dict_morbidities = {
    4: "Acute_Kidney_Injury",
    5: "ARDS",
    8: "Cardiac_Arrest_CPR",
    12: "Deep_Surgical_Site_Infection",
    14: "DVT",
    19: "Organ_Space_Surgical_Site_Infection",
    21: "Pulmonary_Embolism",
    25: "Unplanned_Intubation",
    32: "Severe_Sepsis",
}
# 0 means that the patient has no allowed morbidity, 1 means he has one
for morbidity_i in allowed_morbidities:
    df_trauma[f"morbidity_{dict_morbidities[morbidity_i]}"] = 0
    for col_compl in complkeys:
        df_trauma.loc[
            (df_trauma[col_compl] == morbidity_i),
            f"morbidity_{dict_morbidities[morbidity_i]}"
        ] = 1  # For every comorbidity, we go through all columns and flag them in the right column

## Blunt/Penetration Feature (method_of_injury)
# Mapping ecode (ecode2 has exactly the sames keys so it's useless to do any mapping)

icd_mapping_ecode = pd.read_csv("/pool001/htazi/Trauma/original_data/icd.csv", sep=";")
dict_icd_mapping_ecode = {}
for j in range(len(icd_mapping_ecode.columns)):
    injury_type = icd_mapping_ecode.columns[j]
    print(injury_type)
    ecode_values = icd_mapping_ecode.iloc[0, j].split(' ')
    ecode_values = [x[6:-1].split("\n")[0] if "float" in x else x.split("\n")[0] for x in ecode_values]
    ecode_values = [float(x) if ')' not in x else float(x[:-1]) for x in ecode_values]
    dict_icd_mapping_ecode.update({ecode: injury_type for ecode in ecode_values})
print("===> Preprocessed the ICD mapping")
# We replace all the NaNs with "Unknown"
df_trauma["method_of_injury_ecode"] = df_trauma["ecode"].astype(float).round(1).map(dict_icd_mapping_ecode)
df_trauma["method_of_injury_ecode"] = df_trauma["method_of_injury_ecode"].replace({np.nan: "Unknown"})
print(df_trauma["method_of_injury_ecode"].value_counts())


## Mapping icd10_primary_code
# (icd10_additional_code has exactly the sames keys so it's useless to do any mapping)
icd_mapping_primary = pd.read_csv("/pool001/htazi/Trauma/original_data/icd_primary_ecodes.csv", sep=";")
dict_icd_mapping_primary_icd = {}
for j in range(len(icd_mapping_primary.columns)):
    injury_type = icd_mapping_primary.columns[j]
    print(injury_type)
    primary_icd_values = icd_mapping_primary.iloc[0, j].split(' ')
    primary_icd_values = [x.split("\n")[0] if "\n" in x else x for x in primary_icd_values]
    dict_icd_mapping_primary_icd.update({primary_icd: injury_type for primary_icd in primary_icd_values})
print("===> Preprocessed the ICD mapping primary icd")
# We replace all the NaNs and unknowns (-1, -2) with "Unknown"
dict_icd_mapping_primary_icd.update({-1: "Unknown", -2: "Unknown"})
df_trauma["method_of_injury_icd_primary"] = df_trauma["icd10_primary_ecode"].map(
    dict_icd_mapping_primary_icd
)
df_trauma["method_of_injury_icd_primary"] = df_trauma["method_of_injury_icd_primary"].replace({np.nan: "Unknown"})
print(df_trauma["method_of_injury_icd_primary"].value_counts())


## Creating final method_of_injury from both other columns
# This will be the final method_of_injury column built from the two intermediary columns
df_trauma["method_of_injury"] = df_trauma.method_of_injury_ecode
# Completing ecode with primary_icd10_code
df_trauma.loc[
    (
        (df_trauma.method_of_injury_ecode == "Unknown") &
         (df_trauma.method_of_injury_icd_primary != "Unknown") &
         (df_trauma.method_of_injury_ecode != df_trauma.method_of_injury_icd_primary)
    )
    , "method_of_injury"
] = df_trauma.loc[
    (
        (df_trauma.method_of_injury_ecode == "Unknown") &
         (df_trauma.method_of_injury_icd_primary != "Unknown") &
         (df_trauma.method_of_injury_ecode != df_trauma.method_of_injury_icd_primary)
    )
    , "method_of_injury_icd_primary"
]

# Completing primary_icd10_code with ecode
df_trauma.loc[
    (
        (df_trauma.method_of_injury_ecode != "Unknown") &
        (df_trauma.method_of_injury_icd_primary == "Unknown") &
        (df_trauma.method_of_injury_ecode != df_trauma.method_of_injury_icd_primary)
    )
    , "method_of_injury"
] = df_trauma.loc[
    (
        (df_trauma.method_of_injury_ecode != "Unknown") &
        (df_trauma.method_of_injury_icd_primary == "Unknown") &
        (df_trauma.method_of_injury_ecode != df_trauma.method_of_injury_icd_primary)
    )
    , "method_of_injury_ecode"
]
# Dropping these two intermediary columns
df_trauma.drop(["method_of_injury_ecode", "method_of_injury_icd_primary"], axis=1, inplace=True)
print(df_trauma.method_of_injury.value_counts())


# Alcohol

df_trauma["alcohol"] = df_trauma["alcohol"].map({
    "Yes (confirmed by test [beyond legal limit])": "Alcohol",
    "No (confirmed by test)": "Residual/no alcohol",
    "Yes (confirmed by test [trace levels])": "Residual/no alcohol",
    "No (not tested)": np.nan
})


# # Predots Cleaning (severity)

ais_inputs = pd.read_csv("/pool001/htazi/Trauma/original_data/ais_inputs.csv", sep=";")
columns_ais = list(ais_inputs.columns)[1:]
injury_locations = [x[9:] for x in columns_ais]


for col, location in zip(columns_ais, injury_locations):
    vars()[f"dict_ais_inputs_{location}"] = {}
    ais_temp = ais_inputs.loc[~ais_inputs[col].isnull(), ["AIS_Predots", col]]
    predots = ais_temp.AIS_Predots.tolist()
    severity = ais_temp[col].tolist()
    vars()[f"dict_ais_inputs_{location}"].update({k:v for k,v in zip(predots, severity)})


for location in injury_locations:
    print(location)
    df_temp = df_trauma.copy()
    df_temp[f"{location}_severity"] = np.nan
    for predot_col in predotkeys:
        df_temp[predot_col] = df_temp[predot_col].astype(float)
        df_temp[predot_col] = df_temp[predot_col].map(vars()[f"dict_ais_inputs_{location}"])
    df_temp[f"{location}_severity"] = df_temp[predotkeys].max(axis=1)
    df_trauma[f"{location}_severity"] = df_temp[f"{location}_severity"].copy()
    print(df_trauma[f"{location}_severity"].value_counts()/len(df_trauma)*100)
    print("% of NaNs: ",df_trauma[f"{location}_severity"].isnull().sum()/len(df_trauma)*100)
    print('----------------------------------')

# Taking the maximum severity for a patient over all new severity columns
df_trauma["severity_max"] = df_trauma[[f"{location}_severity" for location in injury_locations]].max(axis=1)


# Transfers Cleaning

print(f"Length before transfer cleaning (keeping only 'No'): {len(df_trauma)}")
df_trauma = df_trauma[df_trauma.transfer == "No"]
print(f"Length after transfer cleaning (keeping only 'No'): {len(df_trauma)}")


# Filtering nan hospdisp
print(f"Length before filtering NaNs from hospdisp: {len(df_trauma)}")
df_trauma = df_trauma[~df_trauma.hospdisp.isnull()]
print(f"Length after filtering NaNs from hospdisp: {len(df_trauma)}")


# Dropping the severity 9 values (mean unknown)
index_severity_9 = df_trauma[df_trauma.severity_max == 9].index.values
df_trauma.drop(index_severity_9, inplace=True)

# Dropping the severity 6 values
index_severity_6 = df_trauma[df_trauma.severity_max == 6].index.values
df_trauma.drop(index_severity_6, inplace=True)
df_trauma.reset_index(inplace=True, drop=True)

# Splitting (time split) between blunt and injury
blunt_injury = ['Blunt - MVT occupant', 'Blunt - Fall', 'Blunt - MVT motorcyclist', 'Blunt - Other',
               'Blunt - MVT Pedal cyclist/pedestrian']
penetrating_injury = ['Penetrating - Gunshot Wound', 'Penetrating - Stab Wound', 'Penetrating - Other/Mixed']
df_blunt = df_trauma[df_trauma.method_of_injury.isin(blunt_injury)].reset_index(drop=True)
df_penetrating = df_trauma[df_trauma.method_of_injury.isin(penetrating_injury)].reset_index(drop=True)
print(f"Length of blunt dataframe: {len(df_blunt)/len(df_trauma)}")
print(f"Length of penetrating dataframe: {len(df_penetrating)/len(df_trauma)}")

# These columns will have to be remapped the other way around when OptImpute has been performed
str_columns_to_map = ["gender", "race1", "acslevel", "signsoflife",
                      "alcohol", "method_of_injury"]
# Ordered:
mapping_acslevel = {"I": 1, "II": 2, "III": 3, "Unknown": -1}
#Non-ordered
mapping_gender = {"Female": 1, "Male": 0}
mapping_race1 = {'Other Race': 0, 'Black or African American': 1, 'White': 2, 'American Indian': 3,
               'Native Hawaiian or Other Pacific Islander': 4, 'Asian': 5}
mapping_signsoflife = {'Unknown':-1, 'Arrived with signs of life': 1, 'Arrived with NO signs of life': 0}
mapping_alcohol = {"Alcohol": 1, "Residual/no alcohol": 0, "Unknown": -1}
mapping_method_of_injury_penetrating = {'Penetrating - Stab Wound': 1, 'Penetrating - Gunshot Wound': 2,
                                       'Penetrating - Other/Mixed':3}
mapping_method_of_injury_blunt = {'Blunt - MVT occupant': 1, 'Blunt - Fall': 2, 'Blunt - MVT motorcyclist': 3,
 'Blunt - Other': 4, 'Blunt - MVT Pedal cyclist/pedestrian': 5}

# With the new morbidity and acslevel deleted on February 1 2020
dict_injury = {0: "blunt", 1: "penetrating"}
for i, df_injury in enumerate([df_blunt, df_penetrating]):
    print(dict_injury[i])
    columns_to_keep = [
        "inc_key", "age", "gender", "race1",
        # "teachsta", # "region",
        # "acslevel",  # "tmode1", # "transfer",
        "signsoflife", "sbp1",  # "sbp2",
        "pulse1",  # "pulse2",
        "oxysat1",  # "oxysat2",
        "temp1", "gcstot1",  # "gcstot2",
        "alcohol", "bleeding_disorder",
        "current_chemotherapy", "congestive_heart_failure",
        "current_smoker", "chronic_renal_failure",
        "history_cva", "diabetes", "disseminated_cancer",
        "copd", "steroid", "cirrhosis", "history_MI",
        "history_pvd", "hypertension_medication",  # "eddisp",
        "method_of_injury",  # new AIS"
        "Head_severity", "Face_severity", "Neck_severity", "Thorax_severity",
        "Abdomen_severity", "Spine_severity",
        "Upper_Extremity_severity", "Lower_Extremity_severity",
        "Pelvis_Perineum_severity", "External_severity", "severity_max",
        "hemorrhage_ctrl_type"
    ]
    for morbidity_i in allowed_morbidities:
        vars()[f"hosp_morbidity_{morbidity_i}"] = df_injury[f"morbidity_{dict_morbidities[morbidity_i]}"]
        print(vars()[f"hosp_morbidity_{morbidity_i}"].value_counts() / len(vars()[f"hosp_morbidity_{morbidity_i}"]))

    # Imputing categorical variables missing values with other categories
    severities = [
        "Head_severity",
        "Face_severity",
        "Neck_severity",
        "Thorax_severity",
        "Abdomen_severity",
        "Spine_severity",
        "Upper_Extremity_severity",
        "Lower_Extremity_severity",
        "Pelvis_Perineum_severity",
        "External_severity",
        "severity_max"
    ]
    for col in severities:
        df_injury[col] = df_injury[col].fillna(0).astype(int)
    # df_injury["acslevel"] = df_injury["acslevel"].fillna("Unknown")
    df_injury["race1"] = df_injury["race1"].fillna("Other Race")
    # Not filling signsoflife because will be optimputed
    # df_injury["signsoflife"] = df_injury["signsoflife"].fillna("Unknown")
    df_injury["alcohol"] = df_injury["alcohol"].fillna("Unknown")

    for col in str_columns_to_map:
        if col == "method_of_injury":
            print(col)
            print(df_injury[col].isnull().sum())
            df_injury[col] = df_injury[col].replace(vars()[f"mapping_{col}_{dict_injury[i]}"]).astype(int)
        else:
            print(col)
            print(df_injury[col].isnull().sum())
            df_injury[col] = df_injury[col].replace(vars()[f"mapping_{col}"])
            try:
                df_injury[col] = df_injury[col].astype(int)
            except:
                continue

    # Saving all the data before imputation (no split yet)
    base_path_non_imputed = "/pool001/htazi/Trauma/data_comorbidities_non_imputed"
    base_path_imputed = "/pool001/htazi/Trauma/data_comorbidities_imputed"
    df_injury = df_injury[columns_to_keep]
    df_injury.to_csv(
        f"{base_path_non_imputed}/{dict_injury[i]}/trauma_X_{dict_injury[i]}.csv",
        index=False
    )
    for morbidity_i in allowed_morbidities:
        vars()[f"hosp_morbidity_{morbidity_i}"].to_csv(
            f"{base_path_non_imputed}/{dict_injury[i]}/trauma_y_morbidity_{morbidity_i}_{dict_injury[i]}.csv",
            index=False
        )
        vars()[f"hosp_morbidity_{morbidity_i}"].to_csv(
            f"{base_path_imputed}/{dict_injury[i]}/trauma_y_morbidity_{morbidity_i}_{dict_injury[i]}.csv",
            index=False
        )
    print(f"Size of data for injury {dict_injury[i]}: {len(df_injury)}")
    print("--------------------------------------------------------")
