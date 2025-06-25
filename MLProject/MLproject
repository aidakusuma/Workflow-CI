name: Weather_Prediction_MLflow_Project

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      data_path: {type: string, default: "weatherdata_preprocessing.csv"}
      test_size: {type: float, default: 0.2}
      random_state: {type: int, default: 42}
    command: "python modelling.py --data_path {data_path} --test_size {test_size} --random_state {random_state}"
  
  train_random_forest:
    parameters:
      data_path: {type: string, default: "weatherdata_preprocessing.csv"}
      n_estimators: {type: int, default: 100}
      max_depth: {type: int, default: 10}
    command: "python modelling.py --model random_forest --data_path {data_path} --n_estimators {n_estimators} --max_depth {max_depth}"
  
  train_linear_regression:
    parameters:
      data_path: {type: string, default: "weatherdata_preprocessing.csv"}
    command: "python modelling.py --model linear_regression --data_path {data_path}"
  
  train_decision_tree:
    parameters:
      data_path: {type: string, default: "weatherdata_preprocessing.csv"}
      max_depth: {type: int, default: 10}
    command: "python modelling.py --model decision_tree --data_path {data_path} --max_depth {max_depth}"
