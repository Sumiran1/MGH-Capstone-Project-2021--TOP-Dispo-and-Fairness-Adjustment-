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
data_path = "/pool001/htazi/Trauma/time_split_per_injury_new_morbidity/penetrating/"
train_X_time = CSV.read(data_path*"trauma_X_train_time_penetrating.csv")
test_X_time = CSV.read(data_path*"trauma_X_test_time_penetrating.csv")
println("Loaded X train and test data with time split for penetrating")

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
inc_keys_train = train_X_time[!, :inc_key]
train_X_time = train_X_time[!, names]
inc_keys_test = test_X_time[!, :inc_key]
test_X_time = test_X_time[!, names]
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
#for col_name in severity_categorical_var
#    train_X_time[!, col_name] = CategoricalArray(train_X_time[!, col_name], ordered=true)
#    test_X_time[!, col_name] = CategoricalArray(test_X_time[!, col_name], ordered=true)
#    println("Transformed column '", col_name, "' to Ordered Categorical")
#end
#categorical!(train_X_time)
#categorical!(test_X_time)
#println("Transformed all other non-ordered Categorical Variables")

n_train = nrow(train_X_time)
X_time = vcat(train_X_time, test_X_time)
n_total = nrow(X_time)

k_neighbors_penetrating = 100
# Fitting OptImpute on train+test set at the same time (no y variable)
lnr_optimpute_train = IAI.ImputationLearner(method=:opt_knn, knn_k=k_neighbors_penetrating, cluster=true)
IAI.fit!(lnr_optimpute_train, X_time)
X_time = IAI.transform(lnr_optimpute_train, X_time)
train_X_time = X_time[1:n_train, :]
train_X_time = hcat(train_X_time, inc_keys_train)
CSV.write("/pool001/htazi/Trauma/imputed_non_processed/penetrating/train_X_time_penetrating_imputed.csv",train_X_time)

test_X_time = X_time[n_train+1:n_total, :]
test_X_time = hcat(test_X_time, inc_keys_test)
CSV.write("/pool001/htazi/Trauma/imputed_non_processed/penetrating/test_X_time_penetrating_imputed.csv",test_X_time)


