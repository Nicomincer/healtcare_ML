from sklearn.svm import SVC 
from sklearn.metrics import classification_report
import pandas as pd 
import numpy as np 
from pathlib import Path 

caminho = Path(Path.home(), "Documentos/healthcare_ML/")
dataset = pd.read_csv(f"{caminho}/dataset_healthcare/healthcare_dataset.csv").drop(["Room Number", "Discharge Date", "Admission Type", "Date of Admission", "Name", "Hospital", "Doctor"], axis=1)
dataset['Gender'] = (dataset["Gender"] == "Female").astype(int)
train, valid, test = np.split(dataset.sample(frac=1), [int(0.6*len(dataset)), int(0.8*len(dataset))])
X_train = dataset[dataset.columns[0:-2]].values
y_train = dataset[dataset.columns[-1]].values
y_test = dataset[dataset.columns[-1]].values
tipos_sanguineos = list(dataset["Blood Type"].unique())
label_sangue = {"O-": 0, "O+": 1, "B-": 2, "AB+": 3, "A+":4, "AB-": 5, "A-": 6, "B+": 7}
label_medical = {'Diabetes': 0,  'Asthma': 1,  'Obesity': 2, 'Arthritis': 3, 'Hypertension':4, 'Cancer': 5}
label_insurance = {'Medicare': 0, 'UnitedHealthcare': 1, 'Aetna': 2, 'Cigna': 3, 'Blue Cross': 4}
label_medication = {'Aspirin': 0, 'Lipitor': 1, 'Penicillin': 2, 'Paracetamol': 3, 'Ibuprofen': 4}
label_results = {'Inconclusive': 0,  'Normal': 1, 'Abnormal': 2}

dataset["Test Results"] = dataset["Test Results"].replace(label_results)
dataset["Medication"] = dataset["Medication"].replace(label_medication)
dataset["Insurance Provider"] = dataset["Insurance Provider"].replace(label_insurance)
dataset["Medical Condition"] = dataset["Medical Condition"].replace(label_medical)
dataset["Blood Type"] = dataset["Blood Type"].replace(label_sangue)

print(dataset.info())





