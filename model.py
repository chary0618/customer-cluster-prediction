import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import joblib
from sklearn.cluster import KMeans

# Loading the Dataset into Dataframe
df = pd. read_csv ("Mall_Customers.csv")

#print(df.info())
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
wcss_list = []
for i in range (1,11) :
    model = KMeans(n_clusters= i, init = "k-means++", random_state=1)
    model. fit(X)
    wcss_list.append(model. inertia_)


plt.plot(range(1,11),wcss_list)
plt.title("Elbow Method Graph" ) 
plt.xlabel("Number of Clusters")
plt.ylabel("wcss List")
plt.show()

# Training the model on our Dataset
model = KMeans(n_clusters = 4, init = "k-means++", random_state = 1)
y_predict = model.fit_predict(X)
print (y_predict)

# Converting the Dataframe X into a numpy array
X_array = X.values

#Plotting the graph of Clusters
plt. scatter (X_array[y_predict == 0,0],X_array[y_predict == 0,1], s = 100, color = "Green")
plt. scatter (X_array[y_predict == 1,0],X_array[y_predict == 1,1], s = 100, color = "Red")
plt. scatter (X_array[y_predict == 2,0],X_array[y_predict == 2,1], s = 100, color = "Yellow")
plt. scatter (X_array[y_predict == 3,0],X_array[y_predict == 3,1], s = 100, color = "Blue")
 
plt.title("customer segmentation graph")
plt.xlabel("Annual income")
plt.ylabel("spending score")
plt.show()
joblib.dump(model,"Model.pkl")
print("Model saved as Model.pkl")