

import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

"""# Load dataset"""



import pandas as pd

df = pd.read_csv('IRIS.csv')




from sklearn.model_selection import train_test_split

x = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

"""# Random Forest Classifier"""

from sklearn.ensemble import RandomForestClassifier #Importing Random Forest Classifier
from sklearn import metrics

clf=RandomForestClassifier(n_estimators=10)  #Creating a random forest with 10 decision trees
clf.fit(x_train, y_train)  #Training our model
y_pred=clf.predict(x_test)  #testing our model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))  #Measuring the accuracy of our model

"""# Application Streamlit"""

import streamlit as st

st.title("Application Streamlit_Iris Dataset")

st.header("Application Streamlit_Iris Dataset")

sepal_length_mean = float(df['sepal_length'].mean())
sepal_length = st.slider("sepal_length", min_value = df['sepal_length'].min(), max_value = df['sepal_length'].max(), value=sepal_length_mean)

sepal_width_mean = float(df['sepal_width'].mean())
sepal_width = st.slider("sepal_width", min_value = df['sepal_width'].min(), max_value = df['sepal_width'].max(), value=sepal_width_mean)

petal_length_mean = float(df['petal_length'].mean())
petal_length = st.slider("petal_length", min_value = df['petal_length'].min(), max_value = df['petal_length'].max(), value=petal_length_mean)

petal_width_mean = float(df['petal_width'].mean())
petal_width = st.slider("petal_width", min_value = df['petal_width'].min(), max_value = df['petal_width'].max(), value=petal_width_mean)

data=pd.DataFrame({ 'sepal_length' : sepal_length,
                    'sepal_width' : sepal_width ,
                     'petal_length': petal_length,
                   'petal_width': petal_width}, [0])


IRIS= st.button('Predict_data')

y1_pred=clf.predict(data)



if IRIS==True :
  y1_pred=clf.predict(data)
  if y1_pred [0]== 'Iris-versicolor':
    st.write('Iris-versicolor')
  elif y1_pred [0]==('Iris-virginica') :
    st.write('Iris-virginica')
  else :
    st.write('Iris-setosa')

