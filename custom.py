import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import seaborn as sns
import matplotlib.pyplot as plt



uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file:
    input_df = pd.read_csv(uploaded_file, sep=";")
    model = pickle.load(open('linmodel.pkl','rb'))
    Experience = []
    Salary = []
    listm = input_df.values.tolist()
    length = len(listm)
    listm.sort()
    for i in range(length):
        s = listm[i]
        val = int(s[0])
        ans = model.predict([[val]])
        #st.write(ans)
        Experience.append(val)
        Salary.append(ans[0])


    list_of_tuples = list(zip(Experience, Salary))
    resultdf = pd.DataFrame(list_of_tuples,
                  columns=['Experience', 'Salary'])
    st.write(resultdf)

    st.write("Current model trend")

    st.line_chart(data=resultdf, width=600, height=200, use_container_width=True)
else: 
    st.write("Please upload a file")














