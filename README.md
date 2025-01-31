# Crop_Yield_Prediction

Crop Yield Prediction
Project Overview
The Crop Yield Prediction project aims to predict soil fertility based on various soil properties and elemental analysis. The project uses machine learning models to classify soil fertility into three categories: 0 (Less Fertile), 1 (Fertile), and 2 (Highly Fertile). The input features include the ratio of various elements in the soil, such as Nitrogen (N), Phosphorous (P), Potassium (K), soil acidity (pH), electrical conductivity (ec), organic carbon (oc), and other micronutrients like Zinc (Zn), Iron (Fe), Copper (Cu), Manganese (Mn), and Boron (B).

The project is developed by Team Rocket, consisting of:

Kavish Paraswar

Swaraj Patil

Neel Sahastrabudhe

Tanishq Thuse

Project Structure
The project is structured as follows:

Data Exploration: The dataset contains 1288 samples with 12 input features and a target variable (fertility). The dataset is explored to understand the distribution of features and the target variable.

Data Preprocessing: The dataset is split into training and validation sets. The data is then used to train various machine learning models.

Dataset used : https://www.kaggle.com/datasets/rahuljaiswalonkaggle/soil-fertility-dataset

Model Training: Several machine learning models are trained, including:

Random Forest Classifier

Gaussian Naive Bayes

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Model Evaluation: The models are evaluated based on their accuracy and classification reports. The Random Forest Classifier achieved the highest accuracy of 94.95% on the validation set.

Data Modification: The dataset is modified to improve model performance, and the modified data is used for further analysis.

Key Features
Input Features:

N: Ratio of Nitrogen (NH4+) content in soil

P: Ratio of Phosphorous (P) content in soil

K: Ratio of Potassium (K) content in soil

ph: Soil acidity (pH)

ec: Electrical conductivity

oc: Organic carbon

S: Sulfur (S)

zn: Zinc (Zn)

fe: Iron (Fe)

cu: Copper (Cu)

Mn: Manganese (Mn)

B: Boron (B)

Output:

Fertility class: 0 (Less Fertile), 1 (Fertile), 2 (Highly Fertile)

Libraries Used
Pandas: For data manipulation and analysis.

NumPy: For numerical computations.

Seaborn: For data visualization.

Matplotlib: For plotting graphs.

Scikit-learn: For machine learning model training and evaluation.

How to Run the Project
Clone the Repository:

bash
Copy
git clone https://github.com/your-repo/crop-yield-prediction.git
cd crop-yield-prediction
Install Dependencies:

bash
Copy
pip install -r requirements.txt
Run the Jupyter Notebook:

bash
Copy
jupyter notebook GFG_Project.ipynb
Follow the Notebook:

The notebook is divided into sections for data exploration, preprocessing, model training, and evaluation. Follow the steps to understand the project flow.

Results
The Random Forest Classifier achieved the highest accuracy of 94.95% on the validation set. The model's performance can be further improved by tuning hyperparameters and experimenting with other machine learning algorithms.

Future Work
Feature Engineering: Explore additional features or transformations that could improve model performance.

Hyperparameter Tuning: Perform grid search or random search to find the best hyperparameters for the models.

Deployment: Deploy the model as a web application or API for real-time predictions.

Conclusion
The Crop Yield Prediction project successfully predicts soil fertility based on various soil properties using machine learning models. The project demonstrates the potential of using machine learning in agriculture to optimize crop yields and improve soil management practices.

Note: This project is suitable for educational purposes and can be extended for real-world applications in agriculture. The dataset and models can be further refined to improve accuracy and robustness.

Team Rocket
Crop Yield Prediction Project
Date: 31/01/2025