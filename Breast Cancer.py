import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# DATA COLLECTION & PROCESSING

# LOADING DATA TO DATAFRAME
breast_cancer_dataset = pd.read_csv("cancer.csv")
print(breast_cancer_dataset)

# NUMBER OF  ROWS AND COLUMNS IN THE DATASET
a = breast_cancer_dataset.shape
print(a)

# GETTING SOME INFORMATION ABOUT THE DATA
breast_cancer_dataset.info()
print(a)

# CHECKING THE MISSING VALUES
b = breast_cancer_dataset.isnull().sum()
print(b)

# STATISTICAL MEASURES ABOUT THE DATA
breast_cancer_dataset.describe()

# CHECKING THE DISTRIBUTION OF TARGET VARIABLE
data = pd.read_csv("cancer.csv")
print(data['diagnosis'].value_counts())

# LABEL THE DATA
data['diagnosis'].value_counts()
label = LabelEncoder()
label.fit(data['diagnosis'])
print(label.classes_)
data['diagnosis_label'] = label.transform(data['diagnosis'])
print(data['diagnosis_label'])

# SEPARATING THE FEATURES AND TARGET
x = data.drop(['id', 'diagnosis', 'Unnamed: 32', 'diagnosis_label'], axis=1)
y = data['diagnosis_label']

# SPLITTING THE DATA INTO TRAINING DATA & TESTING DATA
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
svm = SVC()
# TRAINING THE SVM MODEL BY USING THE TRAINING DATA
svm.fit(x_train, y_train)

# MODEL EVALUATION
# ACCURACY SCORE
pre = svm.predict(x_test)
print(classification_report(y_test, pre))
print(accuracy_score(y_test, pre))

# PREDICTING THE DATA
input_data = (
    12.31, 16.52, 79.19, 470.9, 0.09172, 0.06829, 0.03372, 0.02272, 0.172, 0.05914, 0.2505, 1.025, 1.74, 19.68,
    0.004854,
    0.01819, 0.01826, 0.007965, 0.01386, 0.002304, 14.11, 23.21, 89.71, 611.1, 0.1176, 0.1843, 0.1703, 0.0866, 0.2618,
    0.07609)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = svm.predict(input_data_reshaped)
if prediction[0] == 0:
    print('Benign cells are not harmful : No Cancer')
else:
    print("'Malignant' cells are harmful : Cancer")
