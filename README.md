# Heart Disease Prediction

A machine learning project focused on predicting heart disease using clinical parameters. This repository contains exploratory data analysis, model development, and evaluation of various machine learning algorithms to identify the presence of heart disease.

## 📋 Project Overview

Heart disease remains one of the leading causes of mortality worldwide. This project aims to leverage machine learning techniques to predict the likelihood of heart disease based on clinical and demographic features. By analyzing patterns in patient data, the model provides a tool that could potentially assist healthcare professionals in early detection and intervention.

## 🔍 Repository Contents

- **Heart_Disease_Data_Analysis.ipynb**: Jupyter notebook containing the complete data analysis, preprocessing, model building, and evaluation
- **heart.csv**: The dataset used for analysis containing patient records and clinical parameters
- **README.md**: Project documentation

## 📊 Dataset Information

The dataset contains several parameters which are considered important in predicting heart disease:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Serum cholesterol
- Fasting blood sugar
- Resting electrocardiographic results
- Maximum heart rate achieved
- Exercise induced angina
- ST depression induced by exercise relative to rest
- Slope of the peak exercise ST segment
- Number of major vessels colored by fluoroscopy
- Thalassemia
- Target variable: Presence of heart disease (1 = present, 0 = absent)

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- TensorFlow/Keras

## 📈 Model Development & Results

The project evaluates multiple machine learning algorithms, including:

- Logistic Regression
- Decision Trees
- Random Forest
- Support Vector Machines
- XGBoost
- Neural Networks

The models are compared based on accuracy, precision, recall, F1-score, and ROC-AUC metrics to identify the best performing approach for heart disease prediction.

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook

### Installation

1. Clone the repository:
```bash
git clone https://github.com/anish3565/Heart_disease.git
cd Heart_disease
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow jupyter
```

4. Launch Jupyter Notebook:
```bash
jupyter notebook
```

5. Open and run the analysis notebook:
   Navigate to and open `Heart_Disease_Data_Analysis.ipynb` in the Jupyter interface.

## 🔑 Key Insights

- Identification of the most significant clinical features correlated with heart disease
- Comparative analysis of different machine learning algorithms' performance
- Visualization of feature importance and their relationships with the target variable
- Optimization techniques to improve model performance

## 📚 Future Work

- Integration with web applications for interactive prediction
- Incorporation of additional datasets for more robust models
- Development of interpretable AI approaches for medical context
- Time-series analysis for disease progression prediction

## 📄 License

This project is available under the MIT License.

## 📬 Contact

For questions, feedback, or collaboration opportunities, please contact Anish through GitHub.

## 🔗 References

- UCI Machine Learning Repository: Heart Disease Dataset
- American Heart Association guidelines and research
- Current medical literature on heart disease risk factors
