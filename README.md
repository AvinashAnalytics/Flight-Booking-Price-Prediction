# ✈️ **Flight Booking Price Prediction**  


<div align="center">
  <img src="https://tse1.mm.bing.net/th?id=OIG1.nlMUn8A80XrIy8_TmfAj&pid=ImgGn" alt="Flight Booking Price Prediction" width="500" height="300">
</div>


_"The future belongs to those who believe in the beauty of their data and the power of machine learning."_  
— **Eleanor Roosevelt**  



---

## 📊 **Project Overview**

Welcome to my **Flight Booking Price Prediction** project! As a **Data Analyst**, I have worked on predicting flight ticket prices using historical data and various features. This project applies **machine learning techniques**, including **Exploratory Data Analysis (EDA)**, feature engineering, and model evaluation, to predict flight prices.

Through thorough **data analysis**, I aimed to uncover valuable insights from the dataset and build a robust prediction model. This project is a perfect blend of data analysis and predictive modeling, demonstrating how data can lead to actionable insights.

---

## 📋 **Dataset Information**

The dataset consists of **around 300,000 records** with **11 columns** that I used to train my models. The data captures different attributes of flight bookings:

- **Airline**: The name of the airline company.
- **Flight**: Flight code.
- **Source City**: The city where the flight departs from.
- **Departure Time**: Time when the flight departs.
- **Stops**: The number of stops between the source and destination cities.
- **Arrival Time**: Time when the flight arrives.
- **Destination City**: The city where the flight lands.
- **Class**: Seat class (Economy, Business).
- **Duration**: Time taken for the journey in hours.
- **Days Left**: The number of days left before the flight departs.
- **Price**: The target variable - the ticket price.

---

## 📂 **Project Structure**

Here is the structure of the project:

```plaintext
├── data/                         # Dataset folder
├── notebooks/                     # Jupyter notebooks for analysis and exploration
├── src/                           # Source code (preprocessing, model training, evaluation)
│   ├── data_preprocessing.py      # Data cleaning and preprocessing script
│   ├── model_training.py          # Model training script
│   └── model_evaluation.py        # Model evaluation script
├── requirements.txt               # Required dependencies
├── LICENSE                        # License file
└── README.md                      # Project README file
```

---

## 🔧 **Installation**

### **Prerequisites**  
To run this project, ensure you have Python 3.x installed along with these libraries:

- **Pandas**: For data manipulation.
- **NumPy**: For numerical computations.
- **Matplotlib / Seaborn**: For visualizations.
- **Scikit-learn**: For building machine learning models.
- **Statsmodels**: For statistical analysis and feature selection (VIF).

### **Install Dependencies**  
Clone the repository and install the dependencies by running the following commands:

```bash
git clone https://github.com/AvinashAnalytics/Flight-Booking-Price-Prediction.git
cd Flight-Booking-Price-Prediction
pip install -r requirements.txt
```

---

## 💻 **Usage**

### **Steps to Run the Project**:

1. **Preprocess the Data**:  
   Run the script to clean and preprocess the data.
    ```bash
    python src/data_preprocessing.py
    ```

2. **Train the Models**:  
   Train machine learning models for price prediction, such as **Linear Regression**, **Decision Tree**, and **Random Forest**.
    ```bash
    python src/model_training.py
    ```

3. **Evaluate the Models**:  
   Evaluate the model performance using **RMSE**, **MAPE**, and **R²** metrics.
    ```bash
    python src/model_evaluation.py
    ```

4. **Interactive Analysis**:  
   For exploratory analysis, you can open the Jupyter notebook:
    ```bash
    notebooks/flight_price_prediction.ipynb
    ```

---

## 🔍 **Model Evaluation**

I used various performance metrics to evaluate the model predictions:

| **Model**              | **RMSE**   | **MAPE**  |
|------------------------|------------|-----------|
| **Linear Regression**   | 7259.93    | 34%       |
| **Decision Tree**       | 3620       | 7.7%      |
| **Random Forest**       | 2824       | 7.3%      |

### **Key Insights**:
- **Linear Regression** performed poorly with a high RMSE of **7259.93** and MAPE of **34%**.
- **Decision Tree** outperformed Linear Regression with an RMSE of **3620** and MAPE of **7.7%**.
- **Random Forest** showed the best performance with the lowest RMSE (**2824**) and MAPE (**7.3%**).

---

## 🚀 **Project Workflow**

### **1. Importing Libraries**  
The first step in the project is to import necessary libraries for data manipulation, visualizations, and machine learning.

### **2. Loading the Data**  
I loaded the dataset into a Pandas DataFrame and performed basic data inspection and cleaning. I removed unnecessary columns and handled missing values.

### **3. Data Preprocessing**  
- **Handling Missing Values**: I checked for missing values and cleaned them appropriately.
- **One-Hot Encoding**: Categorical variables like **Airline**, **Source City**, etc., were one-hot encoded for model training.

### **4. Data Visualization**  
- **Price vs Airline**: I visualized the price distribution across different airlines.
- **Price vs Days Left**: I explored how the flight price is related to the number of days left for departure.
- **Ticket Price Range**: A visualization of ticket prices across all flights.
- **Class-wise Price Range**: A comparison of ticket prices between **Economy** and **Business** class.

### **5. Feature Selection**  
- **Correlation Heatmap**: I generated a heatmap to explore correlations between features and the target variable (Price).
- **Variance Inflation Factor (VIF)**: I used VIF to identify multicollinearity in features and dropped the **Stops** column due to high VIF.

### **6. Machine Learning Models**  
I trained three models to predict flight prices:
- **Linear Regression**: A basic model for prediction.
- **Decision Tree Regressor**: A more advanced model for better accuracy.
- **Random Forest Regressor**: An ensemble method that performed the best in this case.

---

## 📜 **License**

This project is licensed under the MIT License.

---

## 📞 **Contact**

If you have any questions or feedback, feel free to reach out to me:

- **Email**: [Avinash Rai](mailto:masteravinashrai@gmail.com)
- **LinkedIn**: [Avinash Rai](https://www.linkedin.com/in/AvinashAnalytics/)
- **GitHub**: [AvinashAnalytics](https://github.com/AvinashAnalytics)

---

## 🌟 **About Me**

I am a **Data Analyst** with a passion for turning raw data into actionable insights. My expertise includes **data cleaning**, **exploratory data analysis**, **data visualization**, and applying **machine learning algorithms** to solve real-world problems. I'm excited about creating predictive models that can drive business decisions and improve processes across various industries.

