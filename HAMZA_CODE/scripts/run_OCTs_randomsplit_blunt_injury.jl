#using Pkg
#Pkg.add("CategoricalArrays")
#Pkg.add("DataFrames")
#Pkg.add("DataFramesMeta")
#Pkg.add("StatsBase")
#Pkg.add("CSV")

using Random, CategoricalArrays, DataFrames, DataFramesMeta, StatsBase, CSV
println("##########################################################################################")
println("##########################################################################################")
println("Starting prediction pipeline for blunt injuries only")
println("##########################################################################################")
println("##########################################################################################")
#data_path = "../Data/imputed_random_split_per_injury/blunt/"

#Loading X for morbidity and mortality
data_path = "/pool001/htazi/Trauma/imputed_random_split_per_injury_without_severity_6/blunt/"
train_X_random_morbidity = CSV.read(data_path*"morbidity/train_X_morbid.csv")
test_X_random_morbidity = CSV.read(data_path*"morbidity/test_X_morbid.csv")
train_X_random_mortality = CSV.read(data_path*"mortality/train_X_mortal.csv")
test_X_random_mortality = CSV.read(data_path*"mortality/test_X_mortal.csv")

# Loading y mortality
train_y_mortality_random = CSV.read(
    data_path*"mortality/train_y_mortal.csv",

)
test_y_mortality_random = CSV.read(
    data_path*"mortality/test_y_mortal.csv",

)
train_y_mortality_random = identity.(convert(Matrix, train_y_mortality_random)[:,2])
test_y_mortality_random = identity.(convert(Matrix, test_y_mortality_random)[:,2])

# Loading y morbidity
train_y_morbidity_random = CSV.read(
    data_path*"morbidity/train_y_morbid.csv",

)
test_y_morbidity_random = CSV.read(
    data_path*"morbidity/test_y_morbid.csv",

)
train_y_morbidity_random = identity.(convert(Matrix, train_y_morbidity_random)[:,2])
test_y_morbidity_random = identity.(convert(Matrix, test_y_morbidity_random)[:,2])
println("Loaded all data with time split")

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
for col_name in severity_categorical_var
    train_X_random_morbidity[!, col_name] = CategoricalArray(train_X_random_morbidity[!, col_name], ordered=true)
    test_X_random_morbidity[!, col_name] = CategoricalArray(test_X_random_morbidity[!, col_name], ordered=true)
    train_X_random_mortality[!, col_name] = CategoricalArray(train_X_random_mortality[!, col_name], ordered=true)
    test_X_random_mortality[!, col_name] = CategoricalArray(test_X_random_mortality[!, col_name], ordered=true)
    println("Transformed column '", col_name, "' to Ordered Categorical")
end
categorical!(train_X_random_morbidity)
categorical!(test_X_random_morbidity)
categorical!(train_X_random_mortality)
categorical!(test_X_random_mortality)
println("Transformed all other non-ordered Categorical Variables")

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
    #:severity_max, :hemorrhage_ctrl_type, :alcohol, :tmode1, :acslevel
]
train_X_random_morbidity = train_X_random_morbidity[!, names]
test_X_random_morbidity = test_X_random_morbidity[!, names]
train_X_random_mortality = train_X_random_mortality[!, names]
test_X_random_mortality = test_X_random_mortality[!, names]
#println("Dropped alcohol variable")
println("##########################################################################################")
println("##########################################################################################")



# Model results output
df_final_results_random_split = DataFrame(id = Any[], seed = Any[],
          depth = Any[], minbucket = Any[], criterion = Any[], cp = Any[],
          auc_train = Float64[], auc_valid = Float64[], auc_test = Float64[])

function oct(train_X, test_X, train_y, test_y, seed, outcome, minbucket)
    id = "seed=$(seed)___outcome=$(outcome)___minbucket=$(minbucket)___injury=blunt"
    outcome = "$(outcome)"
    # Process Data
    Random.seed!(seed)
    @show size(train_X)
    @show size(test_X)


    println("Running OCT for: $(id)")
    # Run Optimal Classification Trees

    # Set of OCT learner and training grid
    oct_lnr = IAI.OptimalTreeClassifier(
        ls_num_tree_restarts=60,
        random_seed=seed,
        treat_unknown_level_missing=true,
        minbucket=minbucket,
        missingdatamode=:separate_class
    )
    grid = IAI.GridSearch(
        oct_lnr,
        max_depth = 8:10,
        criterion = :gini,
    )
    println("Started Fitting the OCT grid search for $(outcome)...")
    IAI.fit!(grid, train_X, train_y, validation_criterion=:auc, sample_weight=:autobalance)
    println("Finished Fitting the OCT grid search for $(outcome)")
    lnr = IAI.get_learner(grid)
    y_pred_proba = IAI.predict_proba(lnr, test_X)
    CSV.write(
        joinpath(@__DIR__, "outputs/y_pred_proba_$(id).csv"),
        y_pred_proba
    )

    println("Chosen Parameters:")
    for (param, val) in grid.best_params
            println("$(param): $(val)")
    end

    train_auc = IAI.score(grid, train_X, train_y, criterion=:auc)
    test_auc = IAI.score(grid, test_X, test_y, criterion=:auc)

    println("OCT-1 Results of tree $(id) ")
    println("--------------------------------")
    println("Max Depth $(id) = ", grid.best_params[:max_depth])
    println("Training AUC $(id) = ", round(100 * train_auc, digits=3), "%")
    println("Testing AUC $(id)  = ", round(100 * test_auc, digits=3), "%")
    println("##########################################################################################")
    println("##########################################################################################")

    lnr = IAI.get_learner(grid)
    IAI.show_in_browser(IAI.ROCCurve(lnr, test_X, test_y))

    # Save the tree
    IAI.write_html(joinpath(@__DIR__, "outputs/$(id)_auc$(round(Int, test_auc * 1000)).html"),
        lnr)
    IAI.write_json(joinpath(@__DIR__, "outputs/$(id)_auc$(round(Int, test_auc * 1000)).json"),
        lnr)
    println("Saved the trees as HTML and JSON")

    return [
        id, seed, grid.best_params[:max_depth], minbucket,
        grid.best_params[:criterion], grid.best_params[:cp],
        train_auc, train_auc, test_auc
    ]
end

minbucket = 100
seed = 1
# Fit OCTs for mortality
oct_results_mort = oct(
    train_X_random_mortality,
    test_X_random_mortality,
    train_y_mortality_random,
    test_y_mortality_random,
    seed,
    "hosp_mortality",
    minbucket
)
push!(df_final_results_random_split, vcat(oct_results_mort))

oct_results_morb = oct(
    train_X_random_morbidity,
    test_X_random_morbidity,
    train_y_morbidity_random,
    test_y_morbidity_random,
    seed,
    "hosp_morbidity",
    minbucket
)
push!(df_final_results_random_split, vcat(oct_results_morb))
println(df_final_results_random_split)
CSV.write("results_random_split_blunt.csv", df_final_results_random_split)


