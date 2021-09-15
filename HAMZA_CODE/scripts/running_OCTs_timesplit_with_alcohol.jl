using Pkg
Pkg.add("CategoricalArrays")
Pkg.add("DataFrames")
Pkg.add("DataFramesMeta")
Pkg.add("StatsBase")
Pkg.add("CSV")

using Random
using CategoricalArrays
using DataFrames
using DataFramesMeta
using StatsBase
using CSV

train_X_time = CSV.read("/pool001/htazi/Trauma/time_split/trauma_X_train_time.csv")
test_X_time = CSV.read("/pool001/htazi/Trauma/time_split/trauma_X_test_time.csv")

train_y_mortality_time = CSV.read("/pool001/htazi/Trauma/time_split/trauma_y_train_mortality_time.csv", header=false)
test_y_mortality_time = CSV.read("/pool001/htazi/Trauma/time_split/trauma_y_test_mortality_time.csv", header=false)
train_y_mortality_time = convert(Matrix, train_y_mortality_time)[:,1]
test_y_mortality_time = convert(Matrix, test_y_mortality_time)[:,1]

train_y_morbidity_time = CSV.read("/pool001/htazi/Trauma/time_split/trauma_y_train_morbidity_time.csv", header=false)
test_y_morbidity_time = CSV.read("/pool001/htazi/Trauma/time_split/trauma_y_test_morbidity_time.csv", header=false)
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
    :Pelvis_Perineum_severity
]
for col_name in severity_categorical_var
    train_X_time[!, col_name] = CategoricalArray(train_X_time[!, col_name], ordered=true)
    test_X_time[!, col_name] = CategoricalArray(test_X_time[!, col_name], ordered=true)
    println("Transformed column '", col_name, "' to Ordered Categorical")
end
categorical!(train_X_time)
categorical!(test_X_time)
println("Transformed all other non-ordered Categorical Variables")





# Model results output
df_final_results = DataFrame(id = Any[], seed = Any[],
          depth = Any[], minbucket = Any[], criterion = Any[], cp = Any[],
          auc_train = Float64[], auc_valid = Float64[], auc_test = Float64[])

function oct(train_X, test_X, train_y, test_y, seed, outcome, minbucket)
    id = "seed=$(seed)___outcome=$(outcome)___minbucket=$(minbucket)"
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
    println("Started Fitting the OCT grid search...")
    IAI.fit!(grid, train_X, train_y, validation_criterion=:auc)
    println("Finished Fitting the OCT grid search")
    lnr = IAI.get_learner(grid)

    println("Chosen Parameters:")
    for (param, val) in grid.best_params
            println("$(param): $(val)")
    end

    train_auc = IAI.score(grid, train_X, train_y, criterion=:auc)
    test_auc = IAI.score(grid, test_X, test_y, criterion=:auc)

    println("OCT-1 Results of tree $(id) ")
    println("----------------------")
    println("Max Depth $(id) = ", grid.best_params[:max_depth])
    println("Training accuracy $(id) = ", round(100 * train_auc, digits=3), "%")
    println("Testing accuracy $(id)  = ", round(100 * test_auc, digits=3), "%")

    lnr = IAI.get_learner(grid)
    #IAI.show_in_browser(IAI.ROCCurve(lnr, test_X, test_y))

    # Save the tree
    IAI.write_json(
        joinpath(@__DIR__, "outputs/$(id)_auc$(round(Int, test_auc * 1000)).json"),
        lnr,
        write_data=false)

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
    train_X_time,
    test_X_time,
    train_y_mortality_time,
    test_y_mortality_time,
    seed,
    "hosp_mortality",
    minbucket
)
push!(df_final_results_time_split, vcat(oct_results_mort))

oct_results_morb = oct(
    train_X_time,
    test_X_time,
    train_y_morbidity_time,
    test_y_morbidity_time,
    seed,
    "hosp_morbidity",
    minbucket
)
push!(df_final_results_time_split, vcat(oct_results_morb))

CSV.write("results_time_split.csv", df_final_results_time_split)
























