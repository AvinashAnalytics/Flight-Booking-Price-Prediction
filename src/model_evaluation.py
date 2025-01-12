# model_evaluation.py

from model_training import evaluate_model
from data_preprocessing import preprocess_data, scale_features
from sklearn.model_selection import train_test_split

def display_evaluation_results(model_name, r2, mae, mape, mse, rmse):
    """Display the evaluation results."""
    print(f"\n{model_name} Evaluation Results:")
    print(f"R^2: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

if __name__ == "__main__":
    file_path = 'C:/Users/Avinash rai/Downloads/Flight_Booking/Flight_Booking.csv'  # Corrected file path

    # Preprocess data
    x, y = preprocess_data(file_path)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Scale the features
    x_train_scaled, x_test_scaled = scale_features(x_train, x_test)

    # Import trained models
    from model_training import train_linear_regression, train_decision_tree, train_random_forest

    lr = train_linear_regression(x_train_scaled, y_train)
    dt = train_decision_tree(x_train_scaled, y_train)
    rf = train_random_forest(x_train_scaled, y_train)

    # Evaluate models
    lr_results = evaluate_model(lr, x_test_scaled, y_test)
    dt_results = evaluate_model(dt, x_test_scaled, y_test)
    rf_results = evaluate_model(rf, x_test_scaled, y_test)

    # Display evaluation results
    display_evaluation_results("Linear Regression", *lr_results)
    display_evaluation_results("Decision Tree", *dt_results)
    display_evaluation_results("Random Forest", *rf_results)
