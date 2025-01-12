# model_training.py

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

def train_linear_regression(x_train, y_train):
    """Train the Linear Regression model."""
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    return lr

def train_decision_tree(x_train, y_train):
    """Train the Decision Tree model."""
    dt = DecisionTreeRegressor()
    dt.fit(x_train, y_train)
    return dt

def train_random_forest(x_train, y_train):
    """Train the Random Forest model."""
    rf = RandomForestRegressor()
    rf.fit(x_train, y_train)
    return rf

def evaluate_model(model, x_test, y_test):
    """Evaluate the model using various metrics."""
    y_pred = model.predict(x_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    
    return r2, mae, mape, mse, rmse

if __name__ == "__main__":
    from data_preprocessing import preprocess_data, scale_features
    file_path = 'C:/Users/Avinash rai/Downloads/Flight_Booking/Flight_Booking.csv'  # Corrected file path

    # Preprocess data
    x, y = preprocess_data(file_path)

    # Split data into training and testing sets
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Scale the features
    from data_preprocessing import scale_features
    x_train_scaled, x_test_scaled = scale_features(x_train, x_test)

    # Train models
    lr = train_linear_regression(x_train_scaled, y_train)
    dt = train_decision_tree(x_train_scaled, y_train)
    rf = train_random_forest(x_train_scaled, y_train)

    # Evaluate models
    print("Linear Regression Evaluation:")
    lr_results = evaluate_model(lr, x_test_scaled, y_test)
    print(lr_results)

    print("\nDecision Tree Evaluation:")
    dt_results = evaluate_model(dt, x_test_scaled, y_test)
    print(dt_results)

    print("\nRandom Forest Evaluation:")
    rf_results = evaluate_model(rf, x_test_scaled, y_test)
    print(rf_results)
