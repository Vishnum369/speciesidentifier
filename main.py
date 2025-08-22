import streamlit as st
import pandas as pd
import numpy as np
from os import path
import pickle #it is used to read .pkl file

#st.title("hello world")
#st.write("good to see you")

#creating a dataframe
df_Data = pd.DataFrame({'column1':[1,2,3,5],'columns2':['a','b','c','d']})
st.write(df_Data) #displaying the dataframe we created
#
st.title("iris dataset")
df_iris = pd.read_csv(path.join("Data","iris.csv"))
st.write(df_iris)
# #filepath = root/Data/iris.csv
#
st.scatter_chart(df_iris[['sepal_length','sepal_width']])
#
st.title("my favorite place")
df_map = pd.DataFrame(np.array([[12.30526781148728, 76.65521781077636]]),columns=["lat","lon"])
st.write(df_map)
st.map(df_map)
# petal_length = st.slider("please choose a petal length",min_value=1,max_value=6)

st.title("Flower species predictor")
petal_length = st.number_input("please choose a petal length between 1.0 to 6.9",
                               placeholder="please enter the petal length",
                               min_value=1.0,max_value=6.9,value=None)
petal_width = st.number_input("please choose a petal width between 0.1 to 2.5",placeholder="please enter the petal length ",min_value=0.1,max_value=2.5,value=None)
sepal_length = st.number_input("please choose a sepal length  between 4.3 to 7.9",placeholder="please enter the petal length",min_value=4.3,max_value=7.9,value=None)
sepal_width = st.number_input("please choose a sepal width between 2.6 to 4.8",placeholder="please enter the petal length ",min_value=2.6,max_value=4.8,value=None)

user_input = pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width]],
                          columns=['sepal_length','sepal_width','petal_length','petal_width'])

#using the .pkl file ,creating an ML model named 'iris_predictor'
model_path = path.join("Model","iris_classifier.pkl")
with open (model_path,'rb') as file:
    iris_predictor = pickle.load(file)


dict_species = {0:'setosa',1:'versicolor',2:'virginica'}

st.write(user_input)
if st.button("predict species"):
    if((petal_length == None) or (petal_width == None) or
        (sepal_length == None) or (sepal_width == None)):
        st.write("please fill all values") # will be executed when any of the value is not entered properly
    else:
        #prediction can be done here. we are expecting a dataframe
        predicted_species = iris_predictor.predict(user_input)
        #predicted species [0] will give us the value in the dataframe
        #we use that values to find the corresponding species from the dictionary ,'dict_species'
        st.write("the species is",dict_species[predicted_species[0]])

