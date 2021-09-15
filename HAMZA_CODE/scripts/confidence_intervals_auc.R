# load packages
library(pROC)
library(dplyr)
rm(list=ls())
setwd("~/Dropbox (MIT)/Trauma - TQIP MGH - MIT/")
path_folder_y_preds_general = "./folder_y_preds/02_03_2020/outputs_general_y_pred_proba_02_03_2020/"
path_folder_y_tests_blunt = "./folder_y_tests/02_03_2020/blunt/"
path_folder_y_tests_penetrating = "./folder_y_tests/02_03_2020/penetrating/"

path_folder_y_preds_comorb = "./folder_y_preds/02_03_2020/outputs_comorb_y_pred_proba_02_03_2020/"
path_folder_y_tests_comorb_blunt = "./folder_y_tests/02_03_2020/comorbidities/blunt/"
path_folder_y_tests_comorb_penetrating= "./folder_y_tests/02_03_2020/comorbidities/penetrating/"
############################## GENERAL DATASETS ##############################
# morbidity penetrating
y_pred_morbid_penetrating <- read.csv(
  paste(path_folder_y_preds_general, "y_pred_proba_seed=1___outcome=hosp_morbidity___minbucket=100___injury=penetrating.csv", sep="")
)[,2]
y_test_morbid_penetrating <- read.csv(
  paste(path_folder_y_tests_penetrating, "test_y_morbid.csv", sep="")
)[,2]
ci_auc_morbid_penetrating <- ci.auc(y_test_morbid_penetrating, y_pred_morbid_penetrating)

# morbidity blunt
y_pred_morbid_blunt <- read.csv(
  paste(path_folder_y_preds_general, "y_pred_proba_seed=1___outcome=hosp_morbidity___minbucket=100___injury=blunt.csv", sep="")
)[,2]
y_test_morbid_blunt <- read.csv(
  paste(path_folder_y_tests_blunt, "test_y_morbid.csv", sep="")
)[,2]
ci_auc_morbid_blunt <- ci.auc(y_test_morbid_blunt, y_pred_morbid_blunt)

# mortality penetrating
y_pred_mortal_penetrating <- read.csv(
  paste(path_folder_y_preds_general, "y_pred_proba_seed=1___outcome=hosp_mortality___minbucket=100___injury=penetrating.csv", sep="")
)[,2]
y_test_mortal_penetrating <- read.csv(
  paste(path_folder_y_tests_penetrating, "test_y_mortal.csv", sep="")
)[,2]
ci_auc_mortal_penetrating <- ci.auc(y_test_mortal_penetrating, y_pred_mortal_penetrating)

# mortality blunt
y_pred_mortal_blunt <- read.csv(
  paste(path_folder_y_preds_general, "y_pred_proba_seed=1___outcome=hosp_mortality___minbucket=100___injury=blunt.csv", sep="")
)[,2]
y_test_mortal_blunt <- read.csv(
  paste(path_folder_y_tests_blunt, "test_y_mortal.csv", sep="")
)[,2]
ci_auc_mortal_blunt <- ci.auc(y_test_mortal_blunt, y_pred_mortal_blunt)


############################## COMORBIDITY DATASETS ############################## 
# morbidity_4_penetrating
y_pred_morbidity_4_penetrating <- read.csv(
  paste(path_folder_y_preds_comorb, "y_pred_proba_seed=1___outcome=hosp_morbidity_4___minbucket=100___injury=penetrating.csv", sep="")
)[,2]
y_test_morbidity_4_penetrating <- read.csv(
  paste(path_folder_y_tests_comorb_penetrating, "test_y_morbid_4.csv", sep="")
)[,2]
ci_auc_morbidity_4_penetrating <- ci.auc(y_test_morbidity_4_penetrating, y_pred_morbidity_4_penetrating)
# morbidity_4_blunt
y_pred_morbidity_4_blunt <- read.csv(
  paste(path_folder_y_preds_comorb, "y_pred_proba_seed=1___outcome=hosp_morbidity_4___minbucket=100___injury=blunt.csv", sep="")
)[,2]
y_test_morbidity_4_blunt <- read.csv(
  paste(path_folder_y_tests_comorb_blunt, "test_y_morbid_4.csv", sep="")
)[,2]
ci_auc_morbidity_4_blunt <- ci.auc(y_test_morbidity_4_blunt, y_pred_morbidity_4_blunt)

# morbidity_5_penetrating
# morbidity_4_penetrating
y_pred_morbidity_5_penetrating <- read.csv(
  paste(path_folder_y_preds_comorb, "y_pred_proba_seed=1___outcome=hosp_morbidity_5___minbucket=100___injury=penetrating.csv", sep="")
)[,2]
y_test_morbidity_5_penetrating <- read.csv(
  paste(path_folder_y_tests_comorb_penetrating, "test_y_morbid_5.csv", sep="")
)[,2]
ci_auc_morbidity_5_penetrating <- ci.auc(y_test_morbidity_5_penetrating, y_pred_morbidity_5_penetrating)
# morbidity_5_blunt
y_pred_morbidity_5_blunt <- read.csv(
  paste(path_folder_y_preds_comorb, "y_pred_proba_seed=1___outcome=hosp_morbidity_5___minbucket=100___injury=blunt.csv", sep="")
)[,2]
y_test_morbidity_5_blunt <- read.csv(
  paste(path_folder_y_tests_comorb_blunt, "test_y_morbid_5.csv", sep="")
)[,2]
ci_auc_morbidity_5_blunt <- ci.auc(y_test_morbidity_5_blunt, y_pred_morbidity_5_blunt)

# morbidity_8_penetrating
y_pred_morbidity_8_penetrating <- read.csv(
  paste(path_folder_y_preds_comorb, "y_pred_proba_seed=1___outcome=hosp_morbidity_8___minbucket=100___injury=penetrating.csv", sep="")
)[,2]
y_test_morbidity_8_penetrating <- read.csv(
  paste(path_folder_y_tests_comorb_penetrating, "test_y_morbid_8.csv", sep="")
)[,2]
ci_auc_morbidity_8_penetrating <- ci.auc(y_test_morbidity_8_penetrating, y_pred_morbidity_8_penetrating)
# morbidity_8_blunt
y_pred_morbidity_8_blunt <- read.csv(
  paste(path_folder_y_preds_comorb, "y_pred_proba_seed=1___outcome=hosp_morbidity_8___minbucket=100___injury=blunt.csv", sep="")
)[,2]
y_test_morbidity_8_blunt <- read.csv(
  paste(path_folder_y_tests_comorb_blunt, "test_y_morbid_8.csv", sep="")
)[,2]
ci_auc_morbidity_8_blunt <- ci.auc(y_test_morbidity_8_blunt, y_pred_morbidity_8_blunt)

# morbidity_12_penetrating
y_pred_morbidity_12_penetrating <- read.csv(
  paste(path_folder_y_preds_comorb, "y_pred_proba_seed=1___outcome=hosp_morbidity_12___minbucket=100___injury=penetrating.csv", sep="")
)[,2]
y_test_morbidity_12_penetrating <- read.csv(
  paste(path_folder_y_tests_comorb_penetrating, "test_y_morbid_12.csv", sep="")
)[,2]
ci_auc_morbidity_12_penetrating <- ci.auc(y_test_morbidity_12_penetrating, y_pred_morbidity_12_penetrating)
# morbidity_12_blunt
y_pred_morbidity_12_blunt <- read.csv(
  paste(path_folder_y_preds_comorb, "y_pred_proba_seed=1___outcome=hosp_morbidity_12___minbucket=100___injury=blunt.csv", sep="")
)[,2]
y_test_morbidity_12_blunt <- read.csv(
  paste(path_folder_y_tests_comorb_blunt, "test_y_morbid_12.csv", sep="")
)[,2]
ci_auc_morbidity_12_blunt <- ci.auc(y_test_morbidity_12_blunt, y_pred_morbidity_12_blunt)

# morbidity_14_penetrating
y_pred_morbidity_14_penetrating <- read.csv(
  paste(path_folder_y_preds_comorb, "y_pred_proba_seed=1___outcome=hosp_morbidity_14___minbucket=100___injury=penetrating.csv", sep="")
)[,2]
y_test_morbidity_14_penetrating <- read.csv(
  paste(path_folder_y_tests_comorb_penetrating, "test_y_morbid_14.csv", sep="")
)[,2]
ci_auc_morbidity_14_penetrating <- ci.auc(y_test_morbidity_14_penetrating, y_pred_morbidity_14_penetrating)
# morbidity_14_blunt
y_pred_morbidity_14_blunt <- read.csv(
  paste(path_folder_y_preds_comorb, "y_pred_proba_seed=1___outcome=hosp_morbidity_14___minbucket=100___injury=blunt.csv", sep="")
)[,2]
y_test_morbidity_14_blunt <- read.csv(
  paste(path_folder_y_tests_comorb_blunt, "test_y_morbid_14.csv", sep="")
)[,2]
ci_auc_morbidity_14_blunt <- ci.auc(y_test_morbidity_14_blunt, y_pred_morbidity_14_blunt)

# morbidity_19_penetrating
y_pred_morbidity_19_penetrating <- read.csv(
  paste(path_folder_y_preds_comorb, "y_pred_proba_seed=1___outcome=hosp_morbidity_19___minbucket=100___injury=penetrating.csv", sep="")
)[,2]
y_test_morbidity_19_penetrating <- read.csv(
  paste(path_folder_y_tests_comorb_penetrating, "test_y_morbid_19.csv", sep="")
)[,2]
ci_auc_morbidity_19_penetrating <- ci.auc(y_test_morbidity_19_penetrating, y_pred_morbidity_19_penetrating)
# morbidity_19_blunt
y_pred_morbidity_19_blunt <- read.csv(
  paste(path_folder_y_preds_comorb, "y_pred_proba_seed=1___outcome=hosp_morbidity_19___minbucket=100___injury=blunt.csv", sep="")
)[,2]
y_test_morbidity_19_blunt <- read.csv(
  paste(path_folder_y_tests_comorb_blunt, "test_y_morbid_19.csv", sep="")
)[,2]
ci_auc_morbidity_19_blunt <- ci.auc(y_test_morbidity_19_blunt, y_pred_morbidity_19_blunt)

# morbidity_21_penetrating
y_pred_morbidity_21_penetrating <- read.csv(
  paste(path_folder_y_preds_comorb, "y_pred_proba_seed=1___outcome=hosp_morbidity_21___minbucket=100___injury=penetrating.csv", sep="")
)[,2]
y_test_morbidity_21_penetrating <- read.csv(
  paste(path_folder_y_tests_comorb_penetrating, "test_y_morbid_21.csv", sep="")
)[,2]
ci_auc_morbidity_21_penetrating <- ci.auc(y_test_morbidity_21_penetrating, y_pred_morbidity_21_penetrating)
# morbidity_21_blunt
y_pred_morbidity_21_blunt <- read.csv(
  paste(path_folder_y_preds_comorb, "y_pred_proba_seed=1___outcome=hosp_morbidity_21___minbucket=100___injury=blunt.csv", sep="")
)[,2]
y_test_morbidity_21_blunt <- read.csv(
  paste(path_folder_y_tests_comorb_blunt, "test_y_morbid_21.csv", sep="")
)[,2]
ci_auc_morbidity_21_blunt <- ci.auc(y_test_morbidity_21_blunt, y_pred_morbidity_21_blunt)

# morbidity_25_penetrating
y_pred_morbidity_25_penetrating <- read.csv(
  paste(path_folder_y_preds_comorb, "y_pred_proba_seed=1___outcome=hosp_morbidity_25___minbucket=100___injury=penetrating.csv", sep="")
)[,2]
y_test_morbidity_25_penetrating <- read.csv(
  paste(path_folder_y_tests_comorb_penetrating, "test_y_morbid_25.csv", sep="")
)[,2]
ci_auc_morbidity_25_penetrating <- ci.auc(y_test_morbidity_25_penetrating, y_pred_morbidity_25_penetrating)
# morbidity_25_blunt
y_pred_morbidity_25_blunt <- read.csv(
  paste(path_folder_y_preds_comorb, "y_pred_proba_seed=1___outcome=hosp_morbidity_25___minbucket=100___injury=blunt.csv", sep="")
)[,2]
y_test_morbidity_25_blunt <- read.csv(
  paste(path_folder_y_tests_comorb_blunt, "test_y_morbid_25.csv", sep="")
)[,2]
ci_auc_morbidity_25_blunt <- ci.auc(y_test_morbidity_25_blunt, y_pred_morbidity_25_blunt)

# morbidity_32_penetrating
y_pred_morbidity_32_penetrating <- read.csv(
  paste(path_folder_y_preds_comorb, "y_pred_proba_seed=1___outcome=hosp_morbidity_32___minbucket=100___injury=penetrating.csv", sep="")
)[,2]
y_test_morbidity_32_penetrating <- read.csv(
  paste(path_folder_y_tests_comorb_penetrating, "test_y_morbid_32.csv", sep="")
)[,2]
ci_auc_morbidity_32_penetrating <- ci.auc(y_test_morbidity_32_penetrating, y_pred_morbidity_32_penetrating)
# morbidity_32_blunt
y_pred_morbidity_32_blunt <- read.csv(
  paste(path_folder_y_preds_comorb, "y_pred_proba_seed=1___outcome=hosp_morbidity_32___minbucket=100___injury=blunt.csv", sep="")
)[,2]
y_test_morbidity_32_blunt <- read.csv(
  paste(path_folder_y_tests_comorb_blunt, "test_y_morbid_32.csv", sep="")
)[,2]
ci_auc_morbidity_32_blunt <- ci.auc(y_test_morbidity_32_blunt, y_pred_morbidity_32_blunt)

df_ci_auc_penetrating <- data.frame(
  penetrating_mortality = ci_auc_mortal_penetrating,
  penetrating_morbidity_composite = ci_auc_morbid_penetrating,
  penetrating_morbidity_4 = ci_auc_morbidity_4_penetrating,
  penetrating_morbidity_5 = ci_auc_morbidity_5_penetrating,
  penetrating_morbidity_8 = ci_auc_morbidity_8_penetrating,
  penetrating_morbidity_12 = ci_auc_morbidity_12_penetrating,
  penetrating_morbidity_14 = ci_auc_morbidity_14_penetrating,
  penetrating_morbidity_19 = ci_auc_morbidity_19_penetrating,
  penetrating_morbidity_21 = ci_auc_morbidity_21_penetrating,
  penetrating_morbidity_25 = ci_auc_morbidity_25_penetrating,
  penetrating_morbidity_32 = ci_auc_morbidity_32_penetrating
)
rownames(df_ci_auc_penetrating) = c("CI Lower Bound", "CI Median", "CI Upper Bound")
write.csv(df_ci_auc_penetrating, "./confidence_interval_auc_penetrating_02_03_2020.csv")

df_ci_auc_blunt <- data.frame(
  blunt_mortality = ci_auc_mortal_blunt,
  blunt_morbidity_composite = ci_auc_morbid_blunt,
  blunt_morbidity_4 = ci_auc_morbidity_4_blunt,
  blunt_morbidity_5 = ci_auc_morbidity_5_blunt,
  blunt_morbidity_8 = ci_auc_morbidity_8_blunt,
  blunt_morbidity_12 = ci_auc_morbidity_12_blunt,
  blunt_morbidity_14 = ci_auc_morbidity_14_blunt,
  blunt_morbidity_19 = ci_auc_morbidity_19_blunt,
  blunt_morbidity_21 = ci_auc_morbidity_21_blunt,
  blunt_morbidity_25 = ci_auc_morbidity_25_blunt,
  blunt_morbidity_32 = ci_auc_morbidity_32_blunt
)
rownames(df_ci_auc_blunt) = c("CI Lower Bound", "CI Median", "CI Upper Bound")
write.csv(df_ci_auc_blunt, "./confidence_interval_auc_blunt_02_03_2020.csv")
