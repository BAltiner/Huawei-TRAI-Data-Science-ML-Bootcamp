# main.py

# imports
from src.utils import *

from src.data_loader import load_data
from src.feature_engineering import feature_engineering

from src.preprocessing import (
    convert_numeric_columns,
    fill_missing_values,
    encode_target,
    clean_data, 
    data_target_split, 
    train_val_test_split, 
    build_preprocessor,
    drop_columns
)
from src.modeling import train_models, build_models, select_best_model
from src.evaluation import evaluate_validation, evaluate_models_on_test


# Main Function
def main():
    # Load
    df = load_data()

    # Cleaning
    df = clean_data(df)
    df = drop_columns(df,columns="customerID")

    # Type Conversion
    df = convert_numeric_columns(df, "TotalCharges")

    df = fill_missing_values(df)

    # feature engineering
    df = feature_engineering(df)

    # encode_target
    df = encode_target(df)

    # data-target split
    X,y = data_target_split(df)

    # train-val-test split
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X,y)

    # numerical & categorical cols
    numeric_cols = get_numeric_cols(X_train)
    categorical_cols = get_categorical_cols(X_train)

    # Build preprocessor
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    #preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)    

    # Models
    models = build_models()

    validation_results = []
    for name, model in models.items():
        # Training
        trained_model = train_models(model,X_train_processed,y_train)

        # Validation
        result = evaluate_validation(trained_model,X_val_processed,y_val)
        result["trained_model"] = trained_model
        result["model_name"] = name 

        validation_results.append(result)

    # Best Model
    best_result = select_best_model(validation_results)
    best_model = best_result["trained_model"]

    # Test
    test_results = evaluate_models_on_test(best_model,X_test_processed,y_test)
    
    # Final Summary
    print("\nValidation Results")

    for result in validation_results:
        print("-"*60)
        print(result["model_name"])
        print(f"Accuracy : {result['accuracy']:.3f}")
        print(f"Precision: {result['precision']:.3f}")
        print(f"Recall   : {result['recall']:.3f}")
        print(f"F1 Score : {result['f1']:.3f}")
    print(best_result)
    print(test_results["classification_report"])
    print(f"Best Model: {best_result['model_name']}")
    print(f"Validation F1: {best_result['f1']:.3f}")
    print(f"Test F1: {test_results['f1']:.3f}")
    print("="*100)
    print("Conclusion:")
    print(f"{best_result['model_name']} model selected because it achieved the highest F1 score.")

# Run
if __name__ == "__main__":
    main()