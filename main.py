import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    # Load the dataset
    df = pd.read_csv('Crop_recommendation.csv')
    
    # Display basic information about the dataset
    print("\nDataset Information:")
    print(df.info())
    
    print("\nFirst few rows of the dataset:")
    print(df.head())
    
    return df

def preprocess_data(df):
    # Separate features and target
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining set shape:", X_train_scaled.shape)
    print("Testing set shape:", X_test_scaled.shape)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, X_test, y_train, y_test):
    # Initialize the model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("\nModel Accuracy:", accuracy)
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return rf_model

def predict_crop(model, scaler):
    print("\n=== Crop Recommendation System ===")
    
    # Get user input
    N = float(input("Enter Nitrogen content in soil (N): "))
    P = float(input("Enter Phosphorus content in soil (P): "))
    K = float(input("Enter Potassium content in soil (K): "))
    temperature = float(input("Enter Temperature (°C): "))
    humidity = float(input("Enter Humidity (%): "))
    ph = float(input("Enter pH value: "))
    rainfall = float(input("Enter Rainfall (mm): "))
    
    # Create input array
    user_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Scale the input
    user_input_scaled = scaler.transform(user_input)
    
    # Make prediction
    prediction = model.predict(user_input_scaled)
    
    print("\nRecommended crop:", prediction[0])

def main():
    # Load the dataset
    data = load_data()
    
    # Preprocess the data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    
    # Train the model
    model = train_model(X_train, X_test, y_train, y_test)
    
    # Start prediction interface
    while True:
        predict_crop(model, scaler)
        
        choice = input("\nWould you like to try another prediction? (yes/no): ")
        if choice.lower() != 'yes':
            break

if __name__ == "__main__":
    main()