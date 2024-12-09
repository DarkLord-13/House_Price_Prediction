# House Price Prediction

This project demonstrates a machine learning approach to predict house prices using the XGBoost regression algorithm. The dataset used is the California Housing dataset.

## Project Overview

The project involves the following steps:
1. **Data Loading**:
    - Load the California Housing dataset.
2. **Data Preprocessing**:
    - Check for missing values and handle them if necessary.
    - Split the dataset into features and labels.
3. **Model Training**:
    - Train the XGBoost regression model on the training data.
4. **Model Evaluation**:
    - Evaluate the model using metrics such as R-squared error and mean absolute error on both training and test data.
5. **Visualization**:
    - Visualize the actual vs. predicted prices to understand the model's performance.
6. **Prediction System**:
    - Create a system to predict house prices based on input features.

## Dependencies

The project requires the following dependencies:
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

## Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/DarkLord-13/Machine-Learning-01.git
    ```

2. Navigate to the project directory:
    ```sh
    cd Machine-Learning-01
    ```

3. Install the required packages:
    ```sh
    pip install numpy pandas matplotlib seaborn scikit-learn xgboost
    ```

## Usage

1. **Load the Data**:
    - Load the California Housing dataset.
    ```python
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    house = pd.DataFrame(housing.data, columns=housing.feature_names)
    house['price'] = pd.Series(housing.target)
    ```

2. **Data Preprocessing**:
    - Check for missing values and split the data into features and labels.
    ```python
    x = house.drop(columns='price', axis=1)
    y = house['price']
    ```

3. **Train the Model**:
    - Split the data into training and test sets, and train the XGBoost regression model.
    ```python
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    model = XGBRegressor()
    model.fit(x_train, y_train)
    ```

4. **Evaluate the Model**:
    - Evaluate the model's performance using R-squared error and mean absolute error.
    ```python
    from sklearn import metrics

    training_data_prediction = model.predict(x_train)
    r2_train = metrics.r2_score(y_train, training_data_prediction)
    mae_train = metrics.mean_absolute_error(y_train, training_data_prediction)

    testing_data_prediction = model.predict(x_test)
    r2_test = metrics.r2_score(y_test, testing_data_prediction)
    mae_test = metrics.mean_absolute_error(y_test, testing_data_prediction)
    ```

5. **Visualize the Results**:
    - Visualize the actual vs. predicted prices.
    ```python
    plt.scatter(y_train, training_data_prediction)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.show()
    ```

6. **Prediction System**:
    - Create a system to predict house prices based on input features.
    ```python
    input_data = (8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23)
    input_data_as_nparray = np.array(input_data)
    input_data_reshaped = input_data_as_nparray.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    print(prediction)
    ```

## License

This project is licensed under the MIT License.
