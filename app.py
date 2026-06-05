import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage,dendrogram,fcluster

import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR=os.path.dirname(os.path.abspath(__file__))

RAW_DIR=os.path.join(BASE_DIR,"data","raw")
CLEAN_DIR=os.path.join(BASE_DIR,"data","cleaned")
MODEL_DIR=os.path.join(BASE_DIR,"models")

os.makedirs(RAW_DIR,exist_ok=True)
os.makedirs(CLEAN_DIR,exist_ok=True)
os.makedirs(MODEL_DIR,exist_ok=True)

st.set_page_config(page_title="Hierarchical Clustering",layout="wide")

st.title("Hierarchical Clustering - Wholesale Customers")

st.header("1. Data Ingestion")

@st.cache_data
def load_data(): return pd.read_csv(os.path.join(RAW_DIR,"Wholesale customers data.csv"))

df=load_data()

st.success("Dataset Loaded Successfully")

st.dataframe(df,use_container_width=True)

st.header("2. Data Cleaning")

strategy=st.selectbox("Missing Value Strategy",["Mean","Median","Most Frequent","Drop Rows"])

df_clean=df.copy()

if strategy=="Drop Rows":
    df_clean=df_clean.dropna()
else:
    fill_map={"Mean":"mean","Median":"median","Most Frequent":"most_frequent"}
    imputer=SimpleImputer(strategy=fill_map[strategy])
    cols=df_clean.select_dtypes(include=np.number).columns
    df_clean[cols]=imputer.fit_transform(df_clean[cols])

st.dataframe(df_clean,use_container_width=True)

if st.button("Save Cleaned Dataset"):
    path=os.path.join(CLEAN_DIR,"wholesale_cleaned.csv")
    df_clean.to_csv(path,index=False)
    st.success("Dataset Saved Successfully")

st.header("3. Load Cleaned Dataset")

files=[f for f in os.listdir(CLEAN_DIR) if f.endswith(".csv")]

if not files:
    st.warning("No cleaned dataset found")
    st.stop()

file=st.selectbox("Select Dataset",files)

data=pd.read_csv(os.path.join(CLEAN_DIR,file))

st.dataframe(data,use_container_width=True)

st.sidebar.header("Model Settings")

number_of_clusters=st.sidebar.slider("Number Of Clusters",2,10,4)

linkage_method=st.sidebar.selectbox("Linkage Method",["ward","complete","average","single"])

features=[
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen"
]

X=data[features]

scaler=StandardScaler()

X_scaled=scaler.fit_transform(X)

st.header("4. Dendrogram")

linked=linkage(X_scaled,method=linkage_method)

fig,ax=plt.subplots(figsize=(12,6))

dendrogram(linked,truncate_mode="level",p=5,ax=ax)

ax.set_title("Hierarchical Clustering Dendrogram")

st.pyplot(fig)

st.header("5. Hierarchical Clustering")

clusters=fcluster(linked,number_of_clusters,criterion="maxclust")

clusters=clusters-1

data["Cluster"]=clusters

silhouette=silhouette_score(X_scaled,clusters)

st.success(f"Silhouette Score : {silhouette:.4f}")

st.header("6. Model Information")

st.subheader("Linkage Method")

st.write(linkage_method)

st.subheader("Number Of Clusters")

st.write(number_of_clusters)

st.header("7. Cluster Distribution")

cluster_counts=data["Cluster"].value_counts().sort_index()

distribution_df=pd.DataFrame({
    "Cluster":cluster_counts.index,
    "Count":cluster_counts.values
})

st.dataframe(distribution_df,use_container_width=True)

fig2,ax2=plt.subplots(figsize=(8,5))

sns.barplot(data=distribution_df,x="Cluster",y="Count",ax=ax2)

ax2.set_title("Cluster Distribution")

st.pyplot(fig2)

st.header("8. Cluster Profile")

cluster_profile=data.groupby("Cluster")[features].mean()

st.dataframe(cluster_profile,use_container_width=True)

st.header("9. Save Model")

model_path=os.path.join(MODEL_DIR,"hierarchical_model.pkl")

with open(model_path,"wb") as f:
    pickle.dump({"linkage_matrix":linked,"scaler":scaler,"features":features},f)

st.success(f"Model Saved At : {model_path}")

st.header("10. Sample Cluster Assignments")

st.dataframe(data[features+["Cluster"]].head(20),use_container_width=True)

st.header("11. Clustered Dataset")

clustered_path=os.path.join(CLEAN_DIR,"wholesale_clustered.csv")

data.to_csv(clustered_path,index=False)

st.success(f"Clustered Dataset Saved At : {clustered_path}")

st.dataframe(data.head(),use_container_width=True)