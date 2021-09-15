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
#data_path = "../Data/imputed_time_split_per_injury_new_morbidity/blunt/"
data_path = "/pool001/htazi/Trauma/imputed_time_split_per_injury_new_morbidity/blunt/"
train_X_time = CSV.read(data_path*"train_X_time_blunt_imputed.csv")
test_X_time = CSV.read(data_path*"test_X_time_blunt_imputed.csv")

train_y_mortality_time = CSV.read(
    data_path*"trauma_y_train_mortality_time_blunt.csv",
    header=false
)
test_y_mortality_time = CSV.read(
    data_path*"trauma_y_test_mortality_time_blunt.csv",
    header=false
)
train_y_mortality_time = convert(Matrix, train_y_mortality_time)[:,1]
test_y_mortality_time = convert(Matrix, test_y_mortality_time)[:,1]

train_y_morbidity_time = CSV.read(
    data_path*"trauma_y_train_morbidity_time_blunt.csv",
    header=false
)
test_y_morbidity_time = CSV.read(
    data_path*"trauma_y_test_morbidity_time_blunt.csv",
    header=false
)
train_y_morbidity_time = convert(Matrix, train_y_morbidity_time)[:,1]
test_y_morbidity_time = convert(Matrix, test_y_morbidity_time)[:,1]
println("Loaded all data with time split")

severity_categorical_var = [
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
    train_X_time[!, col_name] = CategoricalArray(train_X_time[!, col_name], ordered=true)
    test_X_time[!, col_name] = CategoricalArray(test_X_time[!, col_name], ordered=true)
    println("Transformed column '", col_name, "' to Ordered Categorical")
end
categorical!(train_X_time)
categorical!(test_X_time)
println("Transformed all other non-ordered Categorical Variables")

# Dropped tmode1 and hemorrhage_ctrl_type and severity_max (29/11/2019)
names_with_alcohol = [
    :age, :alcohol,:gender, :race1, :acslevel, #:tmode1,
    :signsoflife, :sbp1, :pulse1, :oxysat1, :temp1, :gcstot1, :bleeding_disorder,
    :current_chemotherapy, :congestive_heart_failure, :current_smoker,
    :chronic_renal_failure, :history_cva, :diabetes, :disseminated_cancer,
    :copd, :steroid, :cirrhosis, :history_MI, :history_pvd, :hypertension_medication,
    :method_of_injury, :Face_severity, :Neck_severity, :Thorax_severity, :Abdomen_severity,
    :Spine_severity, :Upper_Extremity_severity, :Lower_Extremity_severity, :Pelvis_Perineum_severity, :External_severity,
    #:severity_max, #:hemorrhage_ctrl_type
]
train_X_time = train_X_time[!, names_with_alcohol]
test_X_time = test_X_time[!, names_with_alcohol]
println("Dropped alcohol, tmode1, hemorrhage_ctrl_type and severity_max variables")


# Model results output
df_final_results_time_split = DataFrame(id = Any[], seed = Any[],
          depth = Any[], minbucket = Any[], criterion = Any[], cp = Any[],
          auc_train = Float64[], auc_valid = Float64[], auc_test = Float64[])

function oct_h(train_X, test_X, train_y, test_y, seed, outcome, minbucket)
    id = "seed=$(seed)___outcome=$(outcome)___minbucket=$(minbucket)___injury=blunt__hyperplanes"
    outcome = "$(outcome)"
    # Process Data
    Random.seed!(seed)
    @show size(train_X)
    @show size(test_X)


    println("Running OCT-H for: $(id)")
    # Run Optimal Classification Trees

    # Set of OCT-H learner and training grid
    oct_h_lnr = IAI.OptimalTreeClassifier(
        ls_num_tree_restarts=60,
        random_seed=seed,
        treat_unknown_level_missing=true,
        minbucket=minbucket,
        missingdatamode=:separate_class,
        hyperplane_config=(sparsity=3,),fast_num_support_restarts=10
    )
    grid = IAI.GridSearch(
        oct_h_lnr,
        max_depth = 5:7,
        criterion = :gini,
    )
    println("Started Fitting the OCT-H grid search for $(outcome)...")
    IAI.fit!(grid, train_X, train_y, validation_criterion=:auc)
    println("Finished Fitting the OCT-H grid search for $(outcome)")
    lnr = IAI.get_learner(grid)

    println("Chosen Parameters:")
    for (param, val) in grid.best_params
            println("$(param): $(val)")
    end

    train_auc = IAI.score(grid, train_X, train_y, criterion=:auc)
    test_auc = IAI.score(grid, test_X, test_y, criterion=:auc)

    println("OCT-H-1 Results of tree $(id) ")
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
# Fit OCT-Hs for mortality
oct_h_results_mort = oct_h(
    train_X_time,
    test_X_time,
    train_y_mortality_time,
    test_y_mortality_time,
    seed,
    "hosp_mortality",
    minbucket
)
push!(df_final_results_time_split, vcat(oct_h_results_mort))

oct_h_results_morb = oct_h(
    train_X_time,
    test_X_time,
    train_y_morbidity_time,
    test_y_morbidity_time,
    seed,
    "hosp_morbidity",
    minbucket
)
push!(df_final_results_time_split, vcat(oct_h_results_morb))
println(df_final_results_time_split)
CSV.write("results_time_split_blunt_with_alcohol_hyperplanes.csv", df_final_results_time_split)


