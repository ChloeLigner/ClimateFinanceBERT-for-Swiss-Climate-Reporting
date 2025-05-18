# ClimateFinanceBERT-for-Swiss-Climate-Reporting

This repository contains the code and data used for my master's thesis on improving the classification and reporting of climate-related Official Development Assistance (ODA). This is an adaptation of the model ClimateFinanceBERT from Malte Toetzke, Florian Egli and Anna Stünzi (2022). 

## Installation

**Python Interpreter**: `Python 3.8`

Install the required packages using:
```bash
pip install -r requirements.txt
````

## Data
- SDC dataset: Shared by the SDC. It was previously translated using the Google Translation API [df_translated.csv](./data/df_translated.csv).
- OECD CRS dataset: Large global ODA dataset from the [OECD CRS database](https://data-explorer.oecd.org/vis?lc=en&fs[0]=Topic%2C1%7CDevelopment%23DEV%23%7COfficial%20Development%20Assistance%20%28ODA%29%23DEV_ODA%23). Only a script is included (`Parquet_CRS_read.py`) — the full `.parquet` file is too large to be uploaded.

## The Model
The model was applied as-is, without retraining or fine-tuning, using weights made available by Toeztke at al. (2022). The following files are directly from Toeztker et al. (2022): [multi_classifyer.py](./code/multiclass/multi_classifyer.py); [relevance_classifyer.py](./code/relevance/relevance_classifyer.py), [helpers_classifyer.py](./code/preprocessing/helpers_classifyer.py), saved_weights_multiclass.pt, saved_weights_relevance.pt; and the following were slightly adapted: [postprocess_descriptives.py](./code/analyses_data/postprocess_after_classification/postprocess_descriptives.py), [postprocess_full_data.py](./code/analyses_data/postprocess_after_classification/postprocess_full_data.py), [classify_projects.py](./code/predict_texts/classify_projects.py), [preprocess_text.py](./code/preprocessing/preprocess_text.py)

## The Data Analysis

The data analysis is strucutres in my Master thesis as follows:
- CRS reported Rio markers: [Rio_bar_plot.ipynb](.\Code\Plots\CRS_data\Rio_bar_plot.ipynb).
- Classifications of SDC projects with ClimateFinanceBERT (based on SDC project descriptions): [Fig1_rio_lines_SDC.ipynb](./code/Plots/Plots/Fig1_rio_lines_SDC.ipynb) & [PieCharts.ipynb](.\Code\Plots\Plots\PieCharts.ipynb)
- Comparison project-by-project between reported Rio markers and ClimateFinanceBERT Classifications: [Comparison_SDC_CRS.ipynb](.\Code\Plots\Plots\Comparison_SDC_CRS.ipynb).
- Comparison of the model outputs based on textual inputs: [Comparison_plots.ipynb](./code/alternative_Analyses/Comparison_plots.ipynb). 
