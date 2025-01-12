# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load dataset from CSV file."""
    df = pd.read_csv(file_path)
    df = df.drop(columns=['Unnamed: 0'])  # Dropping unnecessary columns
    return df

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    df = df.dropna()  # Drop rows with missing values for simplicity
    return df

def encode_categorical_features(df):
    """Apply Label Encoding to categorical columns."""
    le = LabelEncoder()
    categorical_columns = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])
    return df

def feature_selection(df):
    """Select features and drop unwanted columns."""
    df = df.drop(columns=['flight'])  # Drop 'flight' column
    x = df.drop(columns=['price'])
    y = df['price']
    return x, y

def preprocess_data(file_path):
    """Load, clean, and preprocess the data."""
    df = load_data(file_path)
    df = handle_missing_values(df)
    df = encode_categorical_features(df)
    x, y = feature_selection(df)
    return x, y

def scale_features(x_train, x_test):
    """Scale features using StandardScaler."""
    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x_train)
    x_test_scaled = sc.transform(x_test)
    return x_train_scaled, x_test_scaled

if __name__ == "__main__":
    file_path = 'C:/Users/Avinash rai/Downloads/Flight_Booking/Flight_Booking.csv'  # Corrected file path
    x, y = preprocess_data(file_path)
    
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Scale the features
    x_train_scaled, x_test_scaled = scale_features(x_train, x_test)

    print(f"Preprocessing completed. Training data shape: {x_train_scaled.shape}")
