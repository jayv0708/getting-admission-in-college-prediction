# Getting Admission in College Prediction

## Overview
This project predicts the probability of a student getting admitted to a college/university based on their profile. It applies machine learning techniques to analyze factors like GRE scores, TOEFL scores, Undergraduate GPA, and other academic credentials to estimate the chance of admission.

## Dataset
The dataset used for this project is `admission_predict.csv`. The features included in the dataset are:
- **GRE Score**: Graduate Record Examination score (out of 340)
- **TOEFL Score**: Test of English as a Foreign Language score (out of 120)
- **University Rating**: Rating of the university the student wishes to apply to (out of 5)
- **SOP**: Statement of Purpose strength (out of 5.0)
- **LOR**: Letter of Recommendation strength (out of 5.0)
- **CGPA**: Undergraduate Cumulative Grade Point Average (out of 10)
- **Research**: Research experience (1 for True, 0 for False)
- **Chance of Admit**: The predicted probability of admission (Target Variable, ranging from 0.0 to 1.0)

## Project Structure
- `Admission prediction.ipynb`: A Jupyter Notebook containing data exploration, preprocessing, data visualization, and model training.
- `run.py`: A Python script that loads the dataset, trains a **Linear Regression** model using `scikit-learn`, and outputs the model's accuracy (R² Score) along with a few sample predictions.
- `admission_predict.csv`: The dataset file.

## How to Run

### Prerequisites
Make sure you have Python installed. You will also need the following standard data science libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `jupyter` (to open the notebook)

You can install the dependencies using pip:
```bash
pip install numpy pandas scikit-learn jupyter
```

### Running the Python Script
To train the model and see sample predictions, run the `run.py` script from the terminal:
```bash
python run.py
```
This script will output the Model Accuracy (R² Score) on a 20% test split and predict the chances of admission for two sample profiles.

### Exploring the Jupyter Notebook
To view the detailed exploratory data analysis and model evaluation:
```bash
jupyter notebook "Admission prediction.ipynb"
```

## Results
The `run.py` script successfully trains a Linear Regression model, showing the expected probability of admission for different profiles. The notebook delves deeper into the correlations between various features (like CGPA and GRE Score) and the overall chance of admit.
