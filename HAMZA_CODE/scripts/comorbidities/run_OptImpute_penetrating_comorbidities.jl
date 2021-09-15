#using Pkg
#Pkg.add("CategoricalArrays")
#Pkg.add("DataFrames")
#Pkg.add("DataFramesMeta")
#Pkg.add("StatsBase")
#Pkg.add("CSV")

using Random, CategoricalArrays, DataFrames, DataFramesMeta, StatsBase, CSV
println("##########################################################################################")
println("##########################################################################################")
println("Starting OptImpute for penetrating injuries only")
println("##########################################################################################")
println("##########################################################################################")
#data_path = "../Data/time_split_per_injury_new_morbidity/penetrating/"
data_path = "/pool001/htazi/Trauma/data_comorbidities_non_imputed/penetrating/"
X_penetrating = CSV.read(data_path*"trauma_X_penetrating.csv")
println("Loaded X data for penetrating")

# Dropped tmode1 and hemorrhage_ctrl_type and severity_max (29/11/2019)
# and alcohol (09/01/2020) and acslevel (01/02/2020)
names = [
    :age, :gender, :race1, :signsoflife, :sbp1, :pulse1,
    :oxysat1, :temp1, :gcstot1, :bleeding_disorder, :current_chemotherapy,
    :congestive_heart_failure, :current_smoker, :chronic_renal_failure,
    :history_cva, :diabetes, :disseminated_cancer,  :copd, :steroid,
    :cirrhosis, :history_MI, :history_pvd, :hypertension_medication,
    :method_of_injury, :Head_severity, :Face_severity, :Neck_severity, :Thorax_severity, :Abdomen_severity,
    :Spine_severity, :Upper_Extremity_severity, :Lower_Extremity_severity,
    :Pelvis_Perineum_severity, :External_severity,
    #:severity_max, #:hemorrhage_ctrl_type, #:alcohol, #:tmode1, #:acslevel
]
inc_keys_penetrating = X_penetrating[!, :inc_key]
X_penetrating = X_penetrating[!, names]
println("Kept only relevant columns")

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

n_total = nrow(X_penetrating)

k_neighbors_penetrating = 200
# Fitting OptImpute on train+test set at the same time (no y variable)
lnr_optimpute = IAI.ImputationLearner(method=:opt_knn, knn_k=k_neighbors_penetrating, cluster=true)
IAI.fit!(lnr_optimpute, X_penetrating)
X_penetrating = IAI.transform(lnr_optimpute, X_penetrating)
X_penetrating = hcat(X_penetrating, inc_keys_penetrating)
CSV.write("/pool001/htazi/Trauma/imputed_non_processed_comorbidities/penetrating/trauma_X_penetrating_imputed.csv", X_penetrating)
