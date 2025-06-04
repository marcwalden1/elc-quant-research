# Quantitative Research @ EL Capital, Januray - May 2025

This repository contains work from my five‐month quantitative research internship at EL Capital, where I was mentored by Eric Lee (Founder and CIO) as part of their Algorithm Development Program. Its purpose is to provide quantitative researchers with concrete examples of the design and implementation of ES and NQ futures trading models. Additionally, it showcases my work in building end-to-end pipelines that handle large-scale tick data and culminate in fully automated trading strategies, as well as the application of machine learning methods in quantitative research. The [iFVG_Model](https://github.com/marcwalden1/elcapital-quant-research/tree/main/iFVG_Model) folder implements a multi-timeframe, momentum-based trading strategy for NQ futures, while the [ML_Breakout_Models](https://github.com/marcwalden1/elcapital-quant-research/tree/main/ML_Breakout_Models) folder demonstrates how machine learning can identify high-probability consolidation events for breakout trades. This README provides a more detailed overview of both folders below. Since large parts of my code are currently used in production, I have omitted certain components for confidentiality in agreement with my supervisor.


## iFVG_Model folder

This folder contains the implementation of the momentum-based “Inverse Fair Value Gap” strategy designed to capture high-volatility expansions in NQ futures. It is a fully automated, end-to-end pipeline that ingests live tick data and trades in real time (see [iFVG_main.py](https://github.com/marcwalden1/elcapital-quant-research/blob/main/iFVG_Model/iFVG_main.py)). This model aims to capitalize on one-sided, expansionary behavior in favourable market conditions. I have provided a simplified, surface-level overview of the model in [iFVG_Simplified_Overview.pdf](https://github.com/marcwalden1/elcapital-quant-research/blob/main/iFVG_Model/iFVG_Simplified_Overview.pdf). Please note that some details have been omitted.


### Implementation
- [iFVG_main.py](https://github.com/marcwalden1/elcapital-quant-research/blob/main/iFVG_Model/iFVG_main.py): End-to-end automated pipeline that executes the iFVG model on live data.
- [iFVG_live.py](https://github.com/marcwalden1/elcapital-quant-research/blob/main/iFVG_Model/iFVG_live.py): Contains most model functions used in iFVG_main.py for high and low timeframe model criteria.
- [tick_stream_processor.py](https://github.com/marcwalden1/elcapital-quant-research/blob/main/iFVG_Model/tick_stream_processor.py): Efficiently processes and stores tick data in memory using NumPy.
- [trade_manager.py](https://github.com/marcwalden1/elcapital-quant-research/blob/main/iFVG_Model/trade_manager.py): Manages trade execution logic, including order placement, position tracking, and manages risk.
- [iFVG_results.ipynb](https://github.com/marcwalden1/elcapital-quant-research/blob/main/iFVG_Model/iFVG_results.ipynb): Result metrics and visualizations for different variations of the iFVG model.
- [parquet_utils.py](https://github.com/marcwalden1/elcapital-quant-research/blob/main/iFVG_Model/parquet_utils.py): Efficiently loads, cleans, and transforms tick-level parquet data using Polars.
- [sample_data_NQ.parquet](https://github.com/marcwalden1/elcapital-quant-research/blob/main/iFVG_Model/sample_data_NQ.parquet): A sample Parquet file containing a Polars DataFrame with 500 rows of NQ tick data.
  
#### Extra files
These versions don't support live data ingestion, so I would recommend sticking with the more recently updated [iFVG_main.py](https://github.com/marcwalden1/elcapital-quant-research/blob/main/iFVG_Model/iFVG_main.py) file in both backtested and live environments.
- [iFVG_backtest.py](https://github.com/marcwalden1/elcapital-quant-research/blob/main/iFVG_Model/iFVG_backtest.py): Contains most model functions used for backtesting.
- [iFVG_variations.py](https://github.com/marcwalden1/elcapital-quant-research/blob/main/iFVG_Model/iFVG_variations.py): Provides modifications to the standard baseline model.
- [utils_2.py](https://github.com/marcwalden1/elcapital-quant-research/blob/main/iFVG_Model/utils_2.py): Efficiently loads, cleans, and transforms tick-level data.

Below I have displayed the results over 8 months of 2024 from the iFVG model that is currently being used. This model incorporates a dynamic exit strategy based on price and volume with optimized hyperparameters, and a fixed risk of approximately $1000 is assumed per trade. More details can be found in [iFVG_results.ipynb](https://github.com/marcwalden1/elcapital-quant-research/blob/main/iFVG_Model/iFVG_results.ipynb).

<img width="878" alt="iFVG_PnL" src="https://github.com/user-attachments/assets/6d984b95-d3d3-40f8-9333-2751c2f4ae53" />



## ML_Breakout_Models folder

In this section, my purpose is to show how machine learning (ML) can be applied to datasets containing consolidation data to produce profitable trading strategies. To do so, I collected every lower-timeframe consolidation event in ES for several years of data, engineered features, and trained machine learning models to identify which consolidation patterns historically preceded successful breakouts. Based on this, I designed a trading model that capitalizes on trading the breakouts of only those consolidations selected by trained neural networks. While the ML models provided in this repository (namely logistic regression, decision-tree, and K-nearest neighbors) are more simplistic, I hope to showcase how the same methodology can be easily scaled to larger datasets and more complex architectures (e.g., feed-forward neural networks). The live system built on those results is confidential, so here I present a simpler public demo: a backtest of the 9:30 – 10:00 AM opening-range breakout with additional momentum criteria. Although this demo shows a basic range-break strategy, the same codebase can be modified to trade only the specific consolidation events flagged by advanced ML models.


### Implementation

- [consolidations.ipynb](https://github.com/marcwalden1/elcapital-quant-research/blob/main/ML_Breakout_Models/consolidations.ipynb): Trains the dataset collected in data_collection.py to identify which consolidations are preceded by successful breakouts using logistic regression, decision-tree, and K-nearest neighbors.
- [data_collection.py](https://github.com/marcwalden1/elcapital-quant-research/blob/main/ML_Breakout_Models/data_collection.py): Collects a dataset on lower timeframe consolidations throughout several years on ES. The functions defined in this file are used in consolidations.ipynb.
- [simple_breakout.py](https://github.com/marcwalden1/elcapital-quant-research/blob/main/ML_Breakout_Models/simple_breakout.py): Contains the main functions used to trade the 9:30 - 10:00 AM opening range breakout model. This implementation serves as a proxy for the full consolidation‐breakout strategy that would operate on ML-selected consolidation events.
- [model_variations.py](https://github.com/marcwalden1/elcapital-quant-research/blob/main/ML_Breakout_Models/model_variations.py): Provides potential modifications to the standard baseline model.
- [simple_breakout_results.ipynb](https://github.com/marcwalden1/elcapital-quant-research/blob/main/ML_Breakout_Models/simple_breakout_results.ipynb): Provides the results for the simple_breakout model.
- [my_statistics.py](https://github.com/marcwalden1/elcapital-quant-research/blob/main/ML_Breakout_Models/my_statistics.py): Provides functions used in simple_breakout_results to visualize results.
- [utils.py](https://github.com/marcwalden1/elcapital-quant-research/blob/main/ML_Breakout_Models/utils.py): Efficiently loads, cleans, and transforms tick-level data.
- [sample_data_ES.parquet](https://github.com/marcwalden1/elcapital-quant-research/blob/main/ML_Breakout_Models/sample_data_ES.parquet): A sample Parquet file containing a Polars DataFrame with 500 rows of ES tick data.




If you have questions please email me at marcwalden@g.harvard.edu.

