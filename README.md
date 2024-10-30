# 🚗💨 Japan Used Cars Price Prediction 🏎️💸

### 🏷️ Predicting Car Prices in the Japanese Market 📉

---
## 📌 Project Overview
Welcome to the **Japan Used Cars Price Prediction** project! This project leverages data from Japan's largest online used car marketplace to predict **car prices** based on key factors like **brand, mileage, engine capacity, and more**. By implementing various machine learning models, we identify the best approach for accurate predictions. 🎯

📊 **Dataset**: The dataset contains **2318 car entries** with 10 unique features, making it a solid foundation for machine learning models to predict used car prices accurately and effectively.

➡️ **Dataset Source**: [Japan Used Cars Dataset on GitHub](https://raw.githubusercontent.com/dsrscientist/dataset4/main/Japan_used_cars_datasets.csv)

---
## ⚙️ Project Workflow

1. **Data Preprocessing** 🧹
   - Handling missing data
   - Encoding categorical variables
   - Dropping unnecessary columns (like `id`)
2. **Exploratory Data Analysis (EDA)** 📊
   - Heatmaps and distribution plots to understand feature relationships
3. **Model Building and Evaluation** 🧠
   - Trying multiple regression models
   - Choosing the best model based on performance
4. **Prediction & Comparison** 📈
   - Analyzing actual vs. predicted prices
5. **Deployment** 🚀
   - Saving the model for real-world predictions

---

## 💾 Dataset Features
Each car entry has the following attributes:

| Feature           | Description                     |
|-------------------|---------------------------------|
| `price`           | 🚘 **Target** - Car price (in ¥)|
| `mark`            | 🏷️ Car brand                   |
| `model`           | 📌 Car model                   |
| `year`            | 📅 Year of manufacture         |
| `mileage`         | 🛣️ Distance driven             |
| `engine_capacity` | ⚙️ Engine capacity (cc)        |
| `transmission`    | 🕹️ Type of transmission        |
| `drive`           | 🛞 Drivetrain (2WD, 4WD, AWD)  |
| `hand_drive`      | 👨‍✈️ Steering wheel position  |
| `fuel`            | ⛽ Type of fuel                |

---

## 🧹 Data Preprocessing

Ensuring data quality and consistency was crucial. Key preprocessing steps:

1. **Check for Null Values and Duplicates**: Verified that the dataset had no null or duplicate values.
2. **Encoding Categorical Features**: Converted object data types to numerical for easier model training.
3. **Dropping Irrelevant Columns**: Removed the `id` column as it holds no predictive power.

### Encoding Example

```python
# Encoding categorical variables
data.replace({'transmission': {'at': 0, 'mt': 1, 'cvt': 2},
              'drive': {'2wd': 0, '4wd': 1, 'awd': 2},
              'fuel': {'gasoline': 0, 'diesel': 1, 'hybrid': 2, 'lpg': 3, 'cng': 4}}, inplace=True)
```

---

## 🔍 Exploratory Data Analysis (EDA)

Our analysis focused on understanding the relationship between variables:

### 🔥 Correlation Heatmap

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=[10,8])
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt='0.1f')
plt.title("Correlation Heatmap for Used Car Features")
plt.show()
```

### 📉 Feature Distributions

Visualized distributions to understand the spread of each feature:

```python
import seaborn as sns
for column in data.columns:
    sns.histplot(data[column], kde=True)
    plt.show()
```

---

## 🧑‍💻 Model Training

Multiple models were tested to identify the best-performing one:

- **Linear Regression** 🤖
- **Support Vector Regression (SVR)** 🧬
- **Decision Tree Regressor** 🌳
- **Random Forest Regressor** 🌲🌲
- **Gradient Boosting Regressor** 🚀

### Train-Test Split

We split the dataset into training and testing sets to evaluate model performance accurately.

```python
from sklearn.model_selection import train_test_split
X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Model Performance

Each model’s **R² score** was evaluated on the test data:

| Model                  | R² Score (Test Data) |
|------------------------|----------------------|
| Linear Regression      | **0.057**           |
| Support Vector Regressor | -0.028          |
| Decision Tree Regressor | -0.314          |
| Random Forest Regressor | **0.269**        |
| Gradient Boosting Regressor | **0.240**    |

**Best Model**: 🏆 **Linear Regression** was selected as the final model due to its highest performance.

---

## 📈 Prediction Results

Here’s a sample of actual prices compared to predicted prices across the different models:

| Actual Price | Linear_Reg | SVR   | RF_Reg | GB_Reg | DT_Reg |
|--------------|------------|-------|--------|--------|--------|
| 950          | 881.4      | 1009.9| 830.7  | 854.5  | 1100.0 |
| 760          | 942.6      | 1009.7| 794.4  | 909.4  | 660.0  |
| 400          | 773.6      | 997.5 | 645.4  | 835.3  | 1350.0 |
| 950          | 1173.1     | 1009.8| 1107.2 | 1018.5 | 1400.0 |

---

## 🚀 Deployment

The Linear Regression model was saved using Joblib for future predictions:

```python
import joblib
joblib.dump(lr, 'final_LR_Japan_Used_Car_Model.pkl')
```

---

## 📋 How to Use

1. **Clone the Repository** 📥
   ```bash
   git clone https://github.com/your_username/Japan_Used_Cars_Price_Prediction.git
   cd Japan_Used_Cars_Price_Prediction
   ```

2. **Install Required Libraries** 📦
   ```bash
   pip install -r requirements.txt
   ```

3. **Load and Use the Model** 🧩
   - Load `final_LR_Japan_Used_Car_Model.pkl` and run predictions with new data.

---

## 🔑 Conclusion
This project demonstrates how machine learning can transform Japan's used car market by predicting car prices based on historical data. Linear Regression emerged as the most effective model, providing insightful and actionable predictions.

⚙️ With further refinement, this model could potentially offer even more precise price estimates, helping both sellers and buyers make informed decisions in the car resale market.

---

## 💬 Feedback

We’d love your feedback! Feel free to contribute or suggest improvements. 😊

---

Happy Predicting! 🎉
