#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
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

    test_X_time_injury_imputed = pd.read_csv(
        f"/pool001/htazi/Trauma/imputed_non_processed/{injury}/test_X_time_{injury}_imputed.csv"
    )
    train_X_time_injury_imputed = pd.read_csv(
        f"/pool001/htazi/Trauma/imputed_non_processed/{injury}/train_X_time_{injury}_imputed.csv"
    )
    # Replacing systolic blood pressure of less than 60 by -1 (<=> unknown or error)
    train_X_time_injury_imputed.loc[train_X_time_injury_imputed.sbp1 < 60, "sbp1"] = -1
    test_X_time_injury_imputed.loc[test_X_time_injury_imputed.sbp1 < 60, "sbp1"] = -1
    train_X_time_injury_imputed.reset_index(inplace=True, drop=True)
    test_X_time_injury_imputed.reset_index(inplace=True, drop=True)

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
    test_X_time_injury_imputed.columns = new_columns
    train_X_time_injury_imputed.columns = new_columns

    for t_set in ["train", "test"]:
        str_columns_to_map = ["gender", "race1", "signsoflife"]  # ,"alcohol", "acslevel"]
        vars()[f"{t_set}_X_time_injury_imputed"]["method_of_injury"] = vars()[
            f"{t_set}_X_time_injury_imputed"
        ]["method_of_injury"].map(
            vars()[f"mapping_method_of_injury_{injury}_other_way"]
        )

        # Map the other way around the categorical features that have been imputed
        for col_map in str_columns_to_map:
            vars()[f"{t_set}_X_time_injury_imputed"][col_map] = (
                vars()[f"{t_set}_X_time_injury_imputed"][col_map].round(0).astype(int)
            )
            vars()[f"{t_set}_X_time_injury_imputed"][col_map] = vars()[
                f"{t_set}_X_time_injury_imputed"
            ][col_map].map(vars()[f"mapping_{col_map}_other_way"])
        print(f"Preprocessed {t_set} dataset for injury injuries")

    save_time_train_test_split(injury, train_X_time_injury_imputed, test_X_time_injury_imputed)
    save_random_train_test_split(
        injury, train_X_time_injury_imputed, test_X_time_injury_imputed
    )


def save_time_train_test_split(
    injury: str, train_X_time_injury_imputed: pd.DataFrame, test_X_time_injury_imputed: pd.DataFrame
):
    # Time train/test split (saving the inc_keys as well)
    # Saving train set
    base_filepath_timesplit = "/pool001/htazi/Trauma/imputed_time_split_per_injury_new_morbidity"
    inc_keys_train_X_time_injury_imputed = train_X_time_injury_imputed.inc_key
    inc_keys_train_X_time_injury_imputed.to_csv(
        f"{base_filepath_timesplit}/{injury}/inc_keys_train_X_time_{injury}_imputed"
    )
    train_X_time_injury_imputed.drop("inc_key", axis=1).to_csv(
        f"{base_filepath_timesplit}/{injury}/train_X_time_{injury}_imputed.csv"
    )

    # Saving test set
    inc_keys_test_X_time_injury_imputed = test_X_time_injury_imputed.inc_key
    inc_keys_test_X_time_injury_imputed.to_csv(
        f"{base_filepath_timesplit}/{injury}/inc_keys_test_X_time_{injury}_imputed"
    )
    test_X_time_injury_imputed.drop("inc_key", axis=1).to_csv(
        f"{base_filepath_timesplit}/{injury}/test_X_time_{injury}_imputed.csv"
    )


def save_random_train_test_split(
    injury: str,
    train_X_time_injury_imputed: pd.DataFrame,
    test_X_time_injury_imputed: pd.DataFrame,
):
    base_filepath_timesplit = "/pool001/htazi/Trauma/imputed_time_split_per_injury_new_morbidity"
    y_train_mortality_time = pd.read_csv(
        f"{base_filepath_timesplit}/{injury}/trauma_y_train_mortality_time_{injury}.csv",
        header=None,
    )
    y_test_mortality_time = pd.read_csv(
        f"{base_filepath_timesplit}/{injury}/trauma_y_test_mortality_time_{injury}.csv",
        header=None,
    )
    y_train_morbidity_time = pd.read_csv(
        f"{base_filepath_timesplit}/{injury}/trauma_y_train_morbidity_time_{injury}.csv",
        header=None,
    )
    y_test_morbidity_time = pd.read_csv(
        f"{base_filepath_timesplit}/{injury}/trauma_y_test_morbidity_time_{injury}.csv",
        header=None,
    )
    X = pd.concat(
        [train_X_time_injury_imputed, test_X_time_injury_imputed]
    ).reset_index(drop=True)
    y_mortality = pd.concat(
        [y_train_mortality_time, y_test_mortality_time]
    ).reset_index(drop=True)
    y_morbidity = pd.concat(
        [y_train_morbidity_time, y_test_morbidity_time]
    ).reset_index(drop=True)

    # Filter out severity 6
    X["severity_max"] = X[
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
    indices_severity_6_to_drop = X[X.severity_max == 6].index.values
    X.drop(indices_severity_6_to_drop, axis=0, inplace=True)
    y_mortality.drop(indices_severity_6_to_drop, axis=0, inplace=True)
    y_morbidity.drop(indices_severity_6_to_drop, axis=0, inplace=True)
    X.drop("severity_max", axis=1, inplace=True)

    # Reset index
    X.reset_index(inplace=True, drop=True)
    y_mortality.reset_index(inplace=True, drop=True)
    y_morbidity.reset_index(inplace=True, drop=True)

    # Actual random split using sklearn
    train_X_morbid, test_X_morbid, train_y_morbid, test_y_morbid = train_test_split(
        X, y_morbidity, stratify=y_morbidity, random_state=7, train_size=0.8
    )
    train_X_mortal, test_X_mortal, train_y_mortal, test_y_mortal = train_test_split(
        X, y_mortality, stratify=y_mortality, random_state=7, train_size=0.8
    )
    train_y_morbid.columns = ["label"]
    train_y_morbid = train_y_morbid["label"]
    test_y_morbid.columns = ["label"]
    test_y_morbid = test_y_morbid["label"]
    train_y_mortal.columns = ["label"]
    train_y_mortal = train_y_mortal["label"]
    test_y_mortal.columns = ["label"]
    test_y_mortal = test_y_mortal["label"]

    # Saving the inc_keys and deleting column for X dataframes
    data_path_random = f"/pool001/htazi/Trauma/imputed_random_split_per_injury_without_severity_6/{injury}/"
    for filename in ["train_X_morbid", "test_X_morbid"]:
        inc_keys_filename = vars()[filename].inc_key.reset_index()
        inc_keys_filename.to_csv(
            data_path_random + f"morbidity/inc_keys_{filename}.csv", header=True
        )
        vars()[filename].drop("inc_key", axis=1, inplace=True)

    for filename in ["train_X_mortal", "test_X_mortal"]:
        inc_keys_filename = vars()[filename].inc_key.reset_index()
        inc_keys_filename.to_csv(
            data_path_random + f"mortality/inc_keys_{filename}.csv", header=True
        )
        vars()[filename].drop("inc_key", axis=1, inplace=True)

    # Morbidity first
    for filename in [
        "train_X_morbid",
        "test_X_morbid",
        "train_y_morbid",
        "test_y_morbid",
    ]:
        vars()[filename].reset_index(drop=True, inplace=True)
        vars()[filename].to_csv(
            data_path_random + f"morbidity/{filename}.csv", header=True
        )
        print(f"Saved file {filename}")

    # Then Mortality
    for filename in [
        "train_X_mortal",
        "test_X_mortal",
        "train_y_mortal",
        "test_y_mortal",
    ]:
        vars()[filename].reset_index(drop=True, inplace=True)
        vars()[filename].to_csv(
            data_path_random + f"mortality/{filename}.csv", header=True
        )
        print(f"Saved file {filename}")


preprocessing_imputed_data("penetrating")
preprocessing_imputed_data("blunt")
