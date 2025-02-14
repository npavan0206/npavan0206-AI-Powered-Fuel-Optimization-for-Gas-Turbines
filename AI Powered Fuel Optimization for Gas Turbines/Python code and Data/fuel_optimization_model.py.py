import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from tabulate import tabulate  # For table formatting

# Load the dataset
data = pd.read_excel('fuel_data.xlsx')  # Ensure the file is in the same directory or provide the correct path

# Check for any negative or unusual values
print(data.describe())  # Check min, max, mean for each feature
print(data.isnull().sum())  # Check for missing values

# Preprocess data (features and target)
X = data[['RPM', 'Power_Factor', 'Temperature', 'Pressure', 'Engine_Load', 'Throttle_Position']]
y = data['Fuel_Consumption']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Lasso regression model with regularization to avoid extreme predictions
lasso = Lasso(alpha=0.1)  # You can tune alpha for better regularization
lasso.fit(X_train_scaled, y_train)

# Save the model and scaler for future use
joblib.dump(lasso, 'fuel_optimization_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Predict and apply non-negative constraints
def predict_fuel_consumption(rpm, power_factor, temperature, pressure, engine_load, throttle_position):
    new_data = pd.DataFrame({
        'RPM': [rpm],
        'Power_Factor': [power_factor],
        'Temperature': [temperature],
        'Pressure': [pressure],
        'Engine_Load': [engine_load],
        'Throttle_Position': [throttle_position]
    })
    
    # Scale the input data
    new_data_scaled = scaler.transform(new_data)
    
    # Predict fuel consumption in liters per hour
    predicted_fuel_liters = lasso.predict(new_data_scaled)[0]
    
    # Apply non-negative constraint
    predicted_fuel_liters = max(0, predicted_fuel_liters)
    
    return predicted_fuel_liters

# Command-line interface for users
def get_float_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

if __name__ == '__main__':
    print("Welcome to the Fuel Optimization System")
    
    n = int(input("Enter the number of inputs: "))  # Number of inputs

    results = []

    for _ in range(n):
        rpm = get_float_input("Enter RPM: ")
        power_factor = get_float_input("Enter Power Factor: ")
        temperature = get_float_input("Enter Temperature (°C): ")
        pressure = get_float_input("Enter Pressure (bar): ")
        engine_load = get_float_input("Enter Engine Load (%): ")
        throttle_position = get_float_input("Enter Throttle Position (%): ")

        predicted_fuel_liters = predict_fuel_consumption(
            rpm, power_factor, temperature, pressure, engine_load, throttle_position)
        
        # Store inputs and predicted value in liters
        results.append([
            rpm, 
            power_factor, 
            temperature, 
            pressure, 
            engine_load, 
            throttle_position, 
            predicted_fuel_liters
        ])

    # Print results as a table with proper alignment
    print("\nPredicted Fuel Consumption (Liters per Hour):")
    print(tabulate(
        results,
        headers=[
            'RPM', 
            'Power Factor', 
            'Temperature', 
            'Pressure', 
            'Engine Load', 
            'Throttle Position', 
            'Predicted Fuel Consumption (liters/h)'
        ],
        tablefmt='grid',
        numalign="center",  # Align numbers at the center
        stralign="center"   # Align text at the center
    ))

    # Plot various graphs to show feature relationships with fuel consumption
    def plot_graphs_with_predictions():
        # Predict fuel consumption for the test set
        test_predictions = lasso.predict(X_test_scaled)
        test_predictions = np.maximum(0, test_predictions)  # Apply non-negative constraint

        # Convert test data and predictions to a DataFrame for easier handling
        test_results = pd.DataFrame(X_test, columns=X.columns)
        test_results['Actual_Fuel_Consumption'] = y_test.values
        test_results['Predicted_Fuel_Consumption'] = test_predictions

        # Create subplots for visualization
        fig, axs = plt.subplots(3, 2, figsize=(18, 12))

        # RPM vs Fuel Consumption
        sns.lineplot(ax=axs[0, 0], x='RPM', y='Actual_Fuel_Consumption', data=test_results, label='Actual', color='red')
        sns.scatterplot(ax=axs[0, 0], x='RPM', y='Predicted_Fuel_Consumption', data=test_results, label='Predicted', color='blue')
        axs[0, 0].set_title('RPM vs Fuel Consumption (liters)')
        axs[0, 0].set_xlabel('RPM')
        axs[0, 0].set_ylabel('Fuel Consumption (liters)')
        axs[0, 0].legend()

        # Engine Load vs Fuel Consumption
        sns.lineplot(ax=axs[0, 1], x='Engine_Load', y='Actual_Fuel_Consumption', data=test_results, label='Actual', color='blue')
        sns.scatterplot(ax=axs[0, 1], x='Engine_Load', y='Predicted_Fuel_Consumption', data=test_results, label='Predicted', color='red')
        axs[0, 1].set_title('Engine Load vs Fuel Consumption (liters)')
        axs[0, 1].set_xlabel('Engine Load (%)')
        axs[0, 1].set_ylabel('Fuel Consumption (liters)')
        axs[0, 1].legend()

        # Temperature vs Fuel Consumption
        sns.regplot(ax=axs[1, 0], x='Temperature', y='Actual_Fuel_Consumption', data=test_results, label='Actual', color='green', scatter_kws={'alpha': 0.5})
        sns.scatterplot(ax=axs[1, 0], x='Temperature', y='Predicted_Fuel_Consumption', data=test_results, label='Predicted', color='orange')
        axs[1, 0].set_title('Temperature vs Fuel Consumption (liters)')
        axs[1, 0].set_xlabel('Temperature (°C)')
        axs[1, 0].set_ylabel('Fuel Consumption (liters)')
        axs[1, 0].legend()

        # Pressure vs Fuel Consumption
        sns.regplot(ax=axs[1, 1], x='Pressure', y='Actual_Fuel_Consumption', data=test_results, label='Actual', color='purple', scatter_kws={'alpha': 0.5})
        sns.scatterplot(ax=axs[1, 1], x='Pressure', y='Predicted_Fuel_Consumption', data=test_results, label='Predicted', color='cyan')
        axs[1, 1].set_title('Pressure vs Fuel Consumption (liters)')
        axs[1, 1].set_xlabel('Pressure (bar)')
        axs[1, 1].set_ylabel('Fuel Consumption (liters)')
        axs[1, 1].legend()

        # Power Factor vs Fuel Consumption
        sns.scatterplot(ax=axs[2, 0], x='Power_Factor', y='Actual_Fuel_Consumption', data=test_results, label='Actual', color='orange')
        sns.scatterplot(ax=axs[2, 0], x='Power_Factor', y='Predicted_Fuel_Consumption', data=test_results, label='Predicted', color='blue')
        axs[2, 0].set_title('Power Factor vs Fuel Consumption (liters)')
        axs[2, 0].set_xlabel('Power Factor')
        axs[2, 0].set_ylabel('Fuel Consumption (liters)')
        axs[2, 0].legend()

        # Throttle Position vs Fuel Consumption
        sns.lineplot(ax=axs[2, 1], x='Throttle_Position', y='Actual_Fuel_Consumption', data=test_results, label='Actual', color='cyan')
        sns.scatterplot(ax=axs[2, 1], x='Throttle_Position', y='Predicted_Fuel_Consumption', data=test_results, label='Predicted', color='magenta')
        axs[2, 1].set_title('Throttle Position vs Fuel Consumption (liters)')
        axs[2, 1].set_xlabel('Throttle Position (%)')
        axs[2, 1].set_ylabel('Fuel Consumption (liters)')
        axs[2, 1].legend()

        # Adjust layout for better readability
        plt.tight_layout()
        plt.show()

    plot_graphs_with_predictions()
