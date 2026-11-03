!!! This is the scaffold, filled in after our business case coach prompt is run !!!

# Team brief
Business case: README.md, section 1 (merged into dev on [date])
Use case: [who] decides [what] using [the prediction]
Notebook: [file name]
Target: [column] | Problem type: [classification / regression] | ID column: [column / none]
Feature groups: [group]: [columns] -> [transformation] (why)   <- one line per group
Excluded columns: [column] (why)
Model: [family] with [key parameters]
Metric: [name], key in evaluate_model output: ["key"], quality gate: [>= or <=] [threshold]
Split: test [share] and validation [share], both of the full dataset; random_state [number]
Validation rules: [rule] (why it should stop the pipeline)
README follow-ups: [change or gap] -> [owner]
features.py signature: get_feature_preprocessor([params]), settings per group: [e.g. n_bins=4]
Agreed extra parameters: [function(parameter=default)]
utils.py helpers: load_csv, save_csv, save_model, load_model
Environment: [packages beyond pandas, numpy, scikit-learn, joblib, pytest, flake8]
Decisions: [decision] -> [choice] (why)
Owners (team): [module] -> [name] (@[GitHub username]), branch feature/[module-name]   <- one per module, utils first, main last
