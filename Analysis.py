import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
# import seaborn as sns
import plotly.express as px
import streamlit as st
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.simplefilter("ignore")

st.set_page_config(
    page_title="Crop Analysis",
    page_icon="🎋",
    layout="wide"
)

data = pd.read_csv("Crop_recommendation.csv")

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
scale = StandardScaler()
minmax = MinMaxScaler()
data_scale = pd.DataFrame(scale.fit_transform(data.iloc[:,:-1]),columns = data.columns[:-1])
data_scale_min = pd.DataFrame(minmax.fit_transform(data.iloc[:,:-1]),columns = data.columns[:-1])
X = data.drop("label",axis=1)
y = data["label"]
labels = data["label"].unique()
from sklearn.model_selection import train_test_split 
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=.25, random_state = 11)
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
scores = pd.DataFrame(columns = ["Model","Accuracy"])

from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier()
RFC.fit(Xtrain, Ytrain)
print(accuracy_score(Ytest, RFC.predict(Xtest)))
rf_accuracy = accuracy_score(Ytest, RFC.predict(Xtest))
# scores = scores.append({"Model":"Random Forest","Accuracy": accuracy_score(Ytest, RFC.predict(Xtest))*100},ignore_index=True)

# Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(Xtrain, Ytrain)
logistic_accuracy = accuracy_score(Ytest, logistic_model.predict(Xtest))

# K-Means Clustering (Assuming a predefined number of clusters)
from sklearn.metrics import silhouette_score

# K-Means Clustering (Assuming a predefined number of clusters)
kmeans = KMeans(n_clusters=len(data['label'].unique()))
kmeans_clusters = kmeans.fit_predict(data_scale_min)
silhouette_avg = silhouette_score(data_scale_min, kmeans_clusters)
kmeans_accuracy = silhouette_avg  # Using silhouette score as a measure



#creating a header with Header section link
# st.markdown("<h1 style='text-align: center; color: #0b0c0c;'>Crop Recommendation Model</h1>", unsafe_allow_html=True)


# st.sidebar.success("Please select page here")
st.markdown("<h1 style='text-align: center; color: green;'>Crop Recommendation Analysis</h1>", unsafe_allow_html=True)
st.write("This is a simple web application which will help in recommending the type of crop.")
st.divider()
st.write("The application is built using the Random Forest Classifier algorithm.")
st.write("The dataset used for training the model is taken from Kaggle.")
st.write("The dataset contains 22 columns and 2200 rows.")
st.write("This is the first 5 rows of the data.")
st.write(data.head())
st.divider()
st.write("Information about the Data")
st.columns((1,1,1))[1].write(data.dtypes)
st.divider()
st.write("Description of the data.")
st.write(data.describe())
st.divider()

import seaborn as sns

st.write("Checking the outliers of the data.")
col = data.columns

for column in range(len(col)):
    if data[col[column]].dtype != "object": 
        fig, ax = plt.subplots()
        sns.boxplot(data[col[column]],ax=ax)
        # fig.set_size_inches(1,1)
        ax.set_xlabel(col[column], c="r") 
        st.pyplot(fig)

st.write("Checking the distribution of the data.")
for column in range(len(col)):
    if data[col[column]].dtype != "object": 
        fig, ax = plt.subplots()
        sns.distplot(data[col[column]],ax=ax)
        # fig.set_size_inches(1,1)
        ax.set_xlabel(col[column], c="r") 
        st.pyplot(fig)
        
# Create a correlation matrix plot
st.write("Correlation Matrix:")
correlation_matrix = data_scale.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
st.pyplot(plt)

# Display accuracy of all models
st.write("Model Comparisons:")
st.write(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
st.write(f"Logistic Regression Accuracy: {logistic_accuracy * 100:.2f}%")
st.write(f"K-Means Accuracy: {kmeans_accuracy *100:.2f}%")

model_names = ["Random Forest", "Logistic Regression", "K-Means"]
accuracies = [rf_accuracy, logistic_accuracy, silhouette_avg]

fig, ax = plt.subplots()
ax.bar(model_names, accuracies)
ax.set_ylabel("Accuracy")
ax.set_title("Model Comparison")
st.pyplot(fig)
from streamlit_extras.switch_page_button import switch_page
if st.button("Predict the Crop using Random Forest "):
    switch_page("prediction")


