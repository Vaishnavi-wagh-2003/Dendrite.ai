import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

def pprint(obj):
    print(json.dumps(obj, indent=2))

# Load parameters from JSON file
params = json.loads(open("F:/My Downloads/algoparams_from_ui.json").read())

# Display target and feature handling
print("\nTarget:")
target = params['design_state_data']['target']
pprint(target)

print("\nFeature Handling:")
feature_handling = params['design_state_data']['feature_handling']
pprint(feature_handling)

# Load dataset
dataset_name = params['design_state_data']['session_info']['dataset']
print(f"\nDataset name: {dataset_name}")
df = pd.read_csv(dataset_name)

# Data Preprocessing
for col, feature in feature_handling.items():
    if not feature['is_selected']:
        df.drop(col, axis=1, inplace=True)
        continue

    if feature["feature_variable_type"] == "numerical":
        if feature['feature_details']["missing_values"] == "Impute":
            impute_method = feature['feature_details']['impute_with']
            if impute_method == "Average of values":
                df[col].fillna(df[col].mean(), inplace=True)
            elif impute_method == "custom":
                df[col].fillna(feature['feature_details']['impute_value'], inplace=True)
            else:
                raise AssertionError(f"Unknown imputation method: {impute_method}")
    elif feature["feature_variable_type"] == "text":
        labels = {key: num for num, key in enumerate(df[col].unique())}
        df[col] = df[col].apply(lambda x: labels[x])
    else:
        raise AssertionError(f"Unknown feature type: {feature['feature_variable_type']}")

# Feature Reduction
config = params['design_state_data']['feature_reduction']
target_col = target['target']
X = df.drop(target_col, axis=1).values
y = df[target_col].values

if config['feature_reduction_method'] == "Tree-based":
    if target['type'] == "regression":
        sel = SelectFromModel(RandomForestRegressor(
            n_estimators=int(config['num_of_trees']), 
            max_depth=int(config['depth_of_trees']), 
            random_state=42
        ))
    elif target['type'] == "classification":
        sel = SelectFromModel(RandomForestClassifier(
            n_estimators=int(config['num_of_trees']), 
            max_depth=int(config['depth_of_trees']), 
            random_state=42
        ))
    else:
        raise ValueError(f"Unsupported target type: {target['type']}")
    
    sel.fit(X, y)
    feature_importance = sel.estimator_.feature_importances_
    sorted_indices = np.argsort(feature_importance)[::-1]
    keep_columns = list(df.columns[sorted_indices[:int(config['num_of_features_to_keep'])]]) + [target_col]
    df = df[keep_columns]

elif config['feature_reduction_method'] == "No Reduction":
    pass

elif config['feature_reduction_method'] == "Correlation with target":
    corr = df.corr()[target_col].drop(target_col)
    sorted_cor = sorted(dict(abs(corr).items()).items(), key=lambda x: x[1], reverse=True)[:int(config['num_of_features_to_keep'])]
    keep_columns = [key for key, value in sorted_cor] + [target_col]
    df = df[keep_columns]

else:
    raise ValueError(f"Unsupported feature reduction method: {config['feature_reduction_method']}")

print("\nProcessed Dataset:")
print(df.head())
