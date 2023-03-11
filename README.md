_Interactive learning often enhances learner’s understanding of complex concept as it is difficult to fully understand the concept without prior experience. With interactive learning, the learners can see the corresponding outcomes associated with changes in particular conditions and are able to retain and recall information as memory improved. To foster learning in the classroom, this study builds interactive dashboards which enable learners to gain hands-on experiment with statistical time series forecast methods. A time series dataset collected from [Kaggle](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data) containing approximately 3 years of Walmart Sales data (2010-2012) was used for model training. Each dashboard contains model architecture and visualizations illustrating model’s results associated with changes in model’s hyperparameters. Researchers, educators, and students can benefit from using the dashboards because they enhance learning experiences and improve long-term memory as learners have a better understanding of the subject._

<hr />

## Website

- [https://ts-forecast.herokuapp.com/](https://ts-forecast.herokuapp.com/)

## Dataset

- [Raw Data](https://github.com/nphan20181/time_series_forecast/blob/main/data/walmart_sales.zip)
- [Preprocessed Data](https://github.com/nphan20181/time_series_forecast/blob/main/data/ts_dataset.csv)
- [Data Preparation Jupyter Notebook](https://github.com/nphan20181/time_series_forecast/blob/main/prepare_ts_data.ipynb)

## Time Series Analysis

- Stationary & Correlation
- Multiplicative Decomposition

## Time Series Forecast Methods

1. Smoothing Techniques
   1. [Exponential Smoothing](https://github.com/nphan20181/time_series_forecast/blob/main/module/es_model.py)
   1. [Holt's Trend Corrected Exponential Smoothing](https://github.com/nphan20181/time_series_forecast/blob/main/module/holt_trend_es.py)
   1. [Multiplicative Holt-Winters](https://github.com/nphan20181/time_series_forecast/blob/main/module/holt_winters.py)
1. [Autoregressive Integrated Moving Average Exgoneous (ARIMAX)](https://github.com/nphan20181/time_series_forecast/blob/main/module/arima_model.py)
1. [Seasonal Autoregressive Intergrated Moving Average Exgoneous (SARIMAX)](https://github.com/nphan20181/time_series_forecast/blob/main/module/sarimax_model.py)

<hr />

## References

1. Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia. OTexts.com/fpp3. Accessed on February 12, 2023.
1. Mendenhall, W. (2019). SECOND COURSE IN STATISTICS : regression analysis. S.L.: Prentice Hall.
1. Bowerman, B. L., O’Connell, R. T., & Koehler, A. B. (2005). Forecasting, Time Series, and Regression. South-Western Pub.
1. Peixeiro, M. (2022). Time Series Forecasting in Python. Simon and Schuster.
1. Geron, A. (2022). Hands-On Machine Learning With Scikit-Learn, Keras, And Tensorflow 3E. S.L.: O’reilly Uk Limited.

‌
