#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split

# Re-Processing imputed data
def preprocessing_imputed_data(injury: str):
    # These columns will be remapped the other way around when OptImpute has been performed
    str_columns_to_map = [
        "gender",
        "race1",
        #"acslevel",
        "signsoflife",
        # "alcohol",
        "method_of_injury",
    ]
    # Ordered:
    #mapping_acslevel = {"I": 1, "II": 2, "III": 3, "Unknown": -1}
    #mapping_acslevel_other_way = {v: k for k, v in mapping_acslevel.items()}

    # Non-ordered
    mapping_gender = {"Female": 1, "Male": 0}
    mapping_gender_other_way = {v: k for k, v in mapping_gender.items()}

    mapping_race1 = {
        "Other Race": 0,
        "Black or African American": 1,
        "White": 2,
        "American Indian": 3,
        "Native Hawaiian or Other Pacific Islander": 4,
        "Asian": 5,
    }
    mapping_race1_other_way = {v: k for k, v in mapping_race1.items()}

    mapping_signsoflife = {
        "Unknown": -1,
        "Arrived with signs of life": 1,
        "Arrived with NO signs of life": 0,
    }
    mapping_signsoflife_other_way = {v: k for k, v in mapping_signsoflife.items()}

    # mapping_alcohol = {"Alcohol": 1, "Residual/no alcohol": 0, "Unknown": -1}
    # mapping_alcohol_other_way = {v: k for k, v in mapping_alcohol.items()}

    mapping_method_of_injury_penetrating = {
        "Penetrating - Stab Wound": 1,
        "Penetrating - Gunshot Wound": 2,
        "Penetrating - Other/Mixed": 3,
    }
    mapping_method_of_injury_penetrating_other_way = {
        v: k for k, v in mapping_method_of_injury_penetrating.items()
    }

    mapping_method_of_injury_blunt = {
        "Blunt - MVT occupant": 1,
        "Blunt - Fall": 2,
        "Blunt - MVT motorcyclist": 3,
        "Blunt - Other": 4,
        "Blunt - MVT Pedal cyclist/pedestrian": 5,
    }
    mapping_method_of_injury_blunt_other_way = {
        v: k for k, v in mapping_method_of_injury_blunt.items()
    }

    X_injury_imputed = pd.read_csv(
        f"/pool001/htazi/Trauma/imputed_non_processed_comorbidities/{injury}/trauma_X_{injury}_imputed.csv"
    )
    # Replacing systolic blood pressure of less than 60 by -1 (<=> unknown or error)
    X_injury_imputed.loc[X_injury_imputed.sbp1 < 60, "sbp1"] = -1
    X_injury_imputed.reset_index(inplace=True, drop=True)

    # Dropped alcohol on 09/01/2020
    new_columns = [
        "age",
        "gender",
        "race1",
        #"acslevel",
        "signsoflife",
        "sbp1",
        "pulse1",
        "oxysat1",
        "temp1",
        "gcstot1",
        "bleeding_disorder",
        "current_chemotherapy",
        "congestive_heart_failure",
        "current_smoker",
        "chronic_renal_failure",
        "history_cva",
        "diabetes",
        "disseminated_cancer",
        "copd",
        "steroid",
        "cirrhosis",
        "history_MI",
        "history_pvd",
        "hypertension_medication",
        "method_of_injury",
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
        "inc_key",
    ]
    X_injury_imputed.columns = new_columns

    str_columns_to_map = ["gender", "race1", "signsoflife"]  # ,"alcohol", "acslevel"]
    X_injury_imputed["method_of_injury"] = X_injury_imputed["method_of_injury"].map(
        vars()[f"mapping_method_of_injury_{injury}_other_way"]
    )

    # Map the other way around the categorical features that have been imputed
    for col_map in str_columns_to_map:
        X_injury_imputed[col_map] = (
            X_injury_imputed[col_map].round(0).astype(int)
        )
        X_injury_imputed[col_map] = X_injury_imputed[col_map].map(vars()[f"mapping_{col_map}_other_way"])
    print(f"Preprocessed dataset for {injury} injuries")

    save_random_train_test_split(
        injury, X_injury_imputed
    )


def save_random_train_test_split(
    injury: str,
    X_injury_imputed: pd.DataFrame,
):
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
    base_filepath = "/pool001/htazi/Trauma/data_comorbidities_imputed"
    for morbidity_i in allowed_morbidities:
        vars()[f"y_morbidity_{morbidity_i}"] = pd.read_csv(
            f"{base_filepath}/{injury}/trauma_y_morbidity_{morbidity_i}_{injury}.csv",
            header=None,
        )

    # Filter out severity 6
    X_injury_imputed["severity_max"] = X_injury_imputed[
        [
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
        ]
    ].max(axis=1)
    indices_severity_6_to_drop = X_injury_imputed[X_injury_imputed.severity_max == 6].index.values
    X_injury_imputed.drop(indices_severity_6_to_drop, axis=0, inplace=True)
    X_injury_imputed.drop("severity_max", axis=1, inplace=True)
    X_injury_imputed.reset_index(inplace=True, drop=True)
    for morbidity_i in allowed_morbidities:
        vars()[f"y_morbidity_{morbidity_i}"].drop(indices_severity_6_to_drop, axis=0, inplace=True)
        vars()[f"y_morbidity_{morbidity_i}"].reset_index(inplace=True, drop=True)

    for morbidity_i in allowed_morbidities:
        (
            vars()[f"train_X_morbid_{morbidity_i}"],
            vars()[f"test_X_morbid_{morbidity_i}"],
            vars()[f"train_y_morbid_{morbidity_i}"],
            vars()[f"test_y_morbid_{morbidity_i}"]
        ) = train_test_split(
            X_injury_imputed,
            vars()[f"y_morbidity_{morbidity_i}"],
            stratify=vars()[f"y_morbidity_{morbidity_i}"],
            random_state=7,
            train_size=0.8
        )
        vars()[f"train_y_morbid_{morbidity_i}"].columns = ["label"]
        vars()[f"train_y_morbid_{morbidity_i}"] = vars()[f"train_y_morbid_{morbidity_i}"]["label"]
        vars()[f"test_y_morbid_{morbidity_i}"].columns = ["label"]
        vars()[f"test_y_morbid_{morbidity_i}"] = vars()[f"test_y_morbid_{morbidity_i}"]["label"]


        # Saving the inc_keys and deleting column for X dataframes
        data_path_random = f"/pool001/htazi/Trauma/data_comorbidities_imputed/{injury}/"
        for filename in [f"train_X_morbid_{morbidity_i}", f"test_X_morbid_{morbidity_i}"]:
            inc_keys_filename = vars()[filename].inc_key.reset_index()
            inc_keys_filename.to_csv(
                data_path_random + f"/inc_keys/inc_keys_{filename}.csv", header=True
            )
            vars()[filename].drop("inc_key", axis=1, inplace=True)

        for filename in [
            f"train_X_morbid_{morbidity_i}",
            f"test_X_morbid_{morbidity_i}",
            f"train_y_morbid_{morbidity_i}",
            f"test_y_morbid_{morbidity_i}",
        ]:
            vars()[filename].reset_index(drop=True, inplace=True)
            vars()[filename].to_csv(
                data_path_random + f"{filename}.csv", header=True
            )
            print(f"Saved file {filename}")


preprocessing_imputed_data("penetrating")
preprocessing_imputed_data("blunt")
