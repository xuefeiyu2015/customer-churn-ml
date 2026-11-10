# End-to-End Churn Classification Benchmark

## 📚 Description
Customer retention is an important business objective for telecommunications companies because customer churn can lead to lost revenue. Identifying customers at high risk of churn can help retention teams prioritize outreach and allocate limited intervention resources more effectively. This project is to use the widely used IBM Telco Customer Churn dataset, to identify the customers with high churn risk.

## 🚀 Simulated Business Objective 

This project simulates a business scenario: the simulated retention team has the capacity to contact up to 500 customers per month, the model aims to rank customers in descending order of predicted churn risk.

## Dataset

IBM Telco Customer Churn 
[https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv]

Source: Machine Learning Zoomcamp (mlbookcamp-code), Chapter 3

File name: WA_Fn-UseC_-Telco-Customer-Churn.csv
The dataset contans :7043 observations with 21 columns


## How to use this repo
'''
uv sync --locked
'''

## Limitations
- The dataset didn't provide clear feature observation cutoff, reliable prediction horizon and time information to make predictions, the model cannot predict whether the customer can churn within a future time interval.

- High churn risk does not necessarily imply that a customer is likely to respond positively to intervention. The model does not estimate intervention effectivenss, customer lifetime value or intervention costs. 

- Therefore, the predicted churn scores should be treated as decision-support information rather than a direct estimate of expected business value. Evaluating the impact of model-guided retention campaigns would require additional data and experimental validation.

