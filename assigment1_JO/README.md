##Project Setup and Execution Instructions
### Open Visual Studio Code (VS Code).
### Navigate to and open the project folder.
### Open a new terminal within VS Code.
### Create a virtual environment by running the following command:
	python -m venv venv
### Activate the virtual environment:
	venv\Scripts\Activate
### Install all required dependencies by executing:
	pip install -r requirements.txt
### This command runs the requirements.txt file and installs all listed dependencies.
### Execute Part 1:
	python part1/benchmark_part1.py
### Execute Part 2 (Part 1 must be completed beforehand):
	part2/run_part2.py
	python part2/benchmark_part2.py
### Execute Part 3:
	streamlit run part3/app.py



## Part 1: Storing and Retrieving Data

### Objective
The goal of this part is to evaluate whether the dataset should be stored in CSV format
or converted to Parquet, considering performance and scalability.

### Libraries Used
- Pandas: data loading and manipulation
- PyArrow: Parquet storage
- OS / Time: benchmarking

### Benchmark Results (1x scale)
- CSV size: XX MB
- Parquet size: XX MB
- Parquet (Snappy) size: XX MB
- CSV read time: XX seconds
- Parquet read time: XX seconds

### Scalability Analysis (10x and 100x)
Although only the 1x dataset was tested, based on the columnar nature of Parquet
and its compression support, it is expected that Parquet will scale significantly
better than CSV for larger datasets.

### Conclusion
For the current dataset (1x), CSV is usable but Parquet already shows better
performance. For future scaling (10x and 100x), Parquet with Snappy compression
is the recommended storage format.


## Part 2: Data Manipulation, Analysis, and Prediction Models

### Objective
The objective of Part 2 is to analyze the original (1x) stock price dataset,
compare the performance of two dataframe libraries (Pandas and Polars),
enhance the dataset with technical indicators, and build prediction models
to forecast the next-day closing price for all companies.

---

### Dataframe Libraries
Two dataframe libraries were evaluated:

- *Pandas*: Widely used, mature library with extensive ecosystem support.
- *Polars*: A modern, high-performance dataframe library designed for
  efficient execution and better scalability.

Both libraries were used to compute technical indicators and their performance
was benchmarked for comparison.

---

### Technical Indicators
Two technical indicators were added to the dataset using historical price data:

1. *10-day Moving Average (MA-10)*  
   This indicator smooths short-term price fluctuations and helps capture
   local trends in stock prices.

2. *Relative Strength Index (RSI)*  
   RSI is a momentum indicator that measures the magnitude of recent price
   changes to evaluate overbought or oversold conditions.

The indicators were calculated independently per company ticker to ensure
correct time-series behavior.

---

### Target Variable
The prediction target is defined as the *closing price of the next trading day*.
This was created by applying a group-wise shift operation on the closing price
column for each company ticker.

All rows with missing values caused by rolling windows or shifting were removed
to prevent data leakage.

---

### Prediction Models
Two regression models were selected for next-day price prediction:

1. *Linear Regression*  
   Used as a baseline model due to its simplicity, interpretability, and fast
   training time.

2. *Random Forest Regressor*  
   A non-linear ensemble model capable of capturing more complex patterns in
   the data and providing improved predictive performance.

---

### Training and Evaluation
The dataset was split into training and testing sets using an *80-20 split*
without shuffling to preserve the temporal order of the time-series data.

Model performance was evaluated using *Root Mean Squared Error (RMSE)*.
The Random Forest model achieved lower RMSE values, indicating better predictive
performance compared to the linear baseline.

---

### Pandas vs Polars Performance
Benchmarking results showed that Polars executed rolling window computations
and indicator calculations faster than Pandas. While both libraries are suitable
for the current dataset size, Polars demonstrated better performance characteristics
and is expected to scale more efficiently for larger datasets.

---

### Conclusion
Part 2 demonstrates the full pipeline of financial time-series analysis,
from data manipulation and feature engineering to predictive modeling.
The comparison between Pandas and Polars highlights the performance benefits
of modern dataframe libraries, while the modeling results show that non-linear
models are more effective for short-term stock price prediction.


## Part 3: Visualization Dashboard

A visualization dashboard was implemented using Streamlit to display the results
of the prediction models developed in Part 2.

The dashboard allows users to select a company ticker and view actual versus
predicted next-day closing prices. The charts dynamically update based on the
selected company, enabling interactive exploration of model performance.

Streamlit was chosen due to its simplicity, rapid development capabilities,
and strong support for data science workflows.

