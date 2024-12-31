# Flight Booking Price Prediction

## Problem Statement
The objective of this project is to predict flight ticket prices based on various factors like airline, flight number, departure and arrival times, class, and other features. By using the flight booking dataset, we will apply Exploratory Data Analysis (EDA), feature engineering, and Machine Learning algorithms to predict the prices of flights.

## Dataset
The dataset contains approximately 300,000 records and 11 features, including:
- **Airline**: The name of the airline company.
- **Flight**: The flight code.
- **Source City**: City from where the flight departs.
- **Departure Time**: Time of flight departure.
- **Stops**: Number of stops between the source and destination.
- **Arrival Time**: Time of flight arrival.
- **Destination City**: City where the flight lands.
- **Class**: Seat class (Economy, Business, etc.).
- **Duration**: Duration of the flight in hours.
- **Days Left**: The number of days left until departure.
- **Price**: The flight ticket price (target variable).

## Steps Involved

### 1. **Data Manipulation**
   - Import necessary libraries.
   - Load the dataset and remove unnecessary columns.
   - Handle missing data and outliers.

### 2. **Exploratory Data Analysis (EDA)**
   - Visualize the relationships between features and the target variable (`Price`).
   - Perform statistical analysis to understand data distribution.

### 3. **Data Preprocessing**
   - One-hot encode categorical features (e.g., Airline, Source City, etc.).
   - Scale numerical features like `Duration`, `Days Left`, etc.

### 4. **Feature Selection**
   - Identify important features using statistical techniques like VIF (Variance Inflation Factor).
   - Select features that have the highest correlation with the target variable.

### 5. **Machine Learning Algorithms**
   - Implement **Linear Regression**, **Decision Tree Regressor**, and **Random Forest Regressor** models to predict flight prices.
   - Evaluate model performance using metrics like RMSE (Root Mean Squared Error) and MAPE (Mean Absolute Percentage Error).

## Technologies Used
- **Python**: Programming language.
- **Jupyter Notebook**: For interactive coding and visualization.
- **Pandas**: Data manipulation and cleaning.
- **Matplotlib/Seaborn**: Data visualization.
- **Scikit-learn**: Machine learning models and evaluation metrics.
  
## Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/AvinashAnalytics/Flight-Booking-Price-Prediction.git
cd Flight-Booking-Price-Prediction
pip install -r requirements.txt
