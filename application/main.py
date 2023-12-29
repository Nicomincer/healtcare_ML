from sklearn.svm import SVC 
from sklearn.metrics import classification_report
import pandas as pd 
import numpy as np 
from pathlib import Path 

caminho = Path(Path.home(), "Documentos/healtcare_ML")
dataset = pd.read_csv(f"{caminho}/healthcare_dataset.csv").drop(["Room Number", "Discharge Date", "Admission Type", "Date of Admission"], axis=1)
dataset['Gender'] = (dataset["Gender"] == "Female").astype(int)

for valor in range(0, int(dataset["Blood Type"].unique())):
    dataset["Gender"] 

train, valid, test = np.split(dataset.sample(frac=1), [int(0.6*len(dataset)), int(0.8*len(dataset))])

print(dataset["Blood Type"].value_counts())


