# Quantitative Research @ EL Capital, Januray - May 2025

This repository contains work from my five‐month quantitative research internship at EL Capital, where I was mentored by Eric Lee (Founder and CIO) as part of EL Capital’s Algorithm Development Program. Its purpose is to provide quantitative researchers with concrete examples of the design and implementation of ES and NQ futures trading models. Additionally, it showcases my work in building end-to-end pipelines that handle large-scale tick data and culminate in fully automated trading strategies, as well as the application of machine learning methods in quantitative research. The [Momentum_Based_iFVG_Model](https://github.com/marcwalden1/elc-quant-research/tree/main/Momentum_Based_iFVG_Model) folder implements a multi-timeframe, momentum-based trading strategy for NQ futures, while the [ML-based_Consolidation_Breakout_Models](https://github.com/marcwalden1/elc-quant-research/tree/main/ML_Based_Consolidation_Breakout_Models) folder demonstrates how machine learning can identify high-probability consolidation events for breakout trades. This README provides a more detailed overview of both below. Since large parts of my code are currently used in production, I have omitted certain components for confidentiality in agreement with my supervisor.


## Momentum_Based_iFVG_Model [add link here!!!!!!!!]

This subfolder contains the implementation of the momentum-based “Inverse Fair Value Gap” strategy designed to capture high-volatility expansions in NQ futures. It is a fully automated, end-to-end pipeline that ingests live tick data and trades in real time (see iFVG_main.py). [add link here!!!] It aims at capitalizing on one-sided, expansionary behavior in favourable market conditions. I have provided a simplified, surface-level overview of the model in iFVG_Simplified_Overview.pdf [add link here!!!!!]. Please note that some details have been omitted.


### Implementation [add links to all files]
- iFVG_main.py: End-to-end automated pipeline that executes the iFVG model on live data.
- iFVG_live.py: Contains most model functions used in iFVG_main.py for high and low timeframe criteria.
- tick_stream_processor.py: Efficiently processes and stores tick data in memory using NumPy.
- trade_manager.py: Manages trade execution logic, including order placement, position tracking, and manages risk.
- iFVG_results.ipynb: Result metrics and visualizations for different variations of the iFVG model.
- parquet_utils.py: Efficiently loads, cleans, and transforms tick-level parquet data using Polars.
- my_statistics.py: Provides functions used in iFVG_results to visualize results.
- sample_data_NQ.parquet: A sample Parquet file containing a Polars DataFrame with 500 rows of NQ tick data.
  
#### Extra files
These versions don't support live data ingestion, so I would recommend sticking with the more recently updated iFVG_main.py in both backtested and live environments.
- iFVG_backtest.py: Contains most model functions used exclusively for backtesting.
- iFVG_variations.py: Provides modifications to the standard baseline model.
- utils_2.py: Efficiently loads, cleans, and transforms tick-level data


## ML_based_Consolidation_Breakout_Models [add link here!!!!!!!!]

In this section, my purpose is to show how machine learning (ML) can be applied to datasets containing consolidation data to produce profitable trading strategies. To do so, I collected every lower-timeframe consolidation event in ES for several years of data, engineered features, and trained machine learning models to identify which consolidation patterns historically preceded successful breakouts. Based on this, I designed a trading model that capitalizes on the consolidations selected by neural networks. While the ML models provided in this repository (namely logistic regression, decision-tree, and K-nearest neighbors) are more simplistic, I hope to showcase how the same methodology can be easily scaled to larger datasets and more complex architectures (e.g., feed-forward neural networks). The live system built on those results is confidential, so here I present a simpler public demo: a backtest of the 9:30 – 10:00 AM opening-range breakout with additional momentum criteria. Although this demo shows a basic range-break strategy, the same codebase can be modified to trade only the specific consolidation events flagged by advanced ML models.


### Implementation

- consolidations.ipynb: Trains the dataset collected in data_collection.py to identify which consolidations are preceded by successful breakouts that can be traded using logistic regression, decision-tree, and K-nearest neighbors. [MISSING THIS !!!!!!!!!!!!!!!]
- data_collection.py: Defines, identifies, and collects a dataset on lower timeframe consolidations throughout several years on ES. The functions defined in this file are used in consolidations.ipynb.
- simple_breakout.py: Contains the main functions used to trade the 9:30 - 10:00 AM opening range breakout model on the 5min chart. This implementation serves as a proxy for the full consolidation‐breakout strategy that would operate on ML-selected consolidation events.
- model_variations.py: Provides modifications to the standard baseline model.
- simple_breakout_results.ipynb: Provides the results for the simple_breakout model.
- my_statistics.py: Provides functions used in simple_breakout_results to visualize results.
- utils.py: Efficiently loads, cleans, and transforms tick-level data
- sample_data_ES.parquet: A sample Parquet file containing a Polars DataFrame with 500 rows of ES tick data.




If you have questions please email me at marcwalden@g.harvard.edu.

