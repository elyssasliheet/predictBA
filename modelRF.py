# Use PQRs to generate Coulombic energies computed with treecode as labels
# train the model with e-features, test accuracy.

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from numpy import savetxt
import os 
import re
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D, AveragePooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.initializers import GlorotUniform
from keras import regularizers
from keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # to apply PCA
from sklearn.model_selection import KFold
import sys
from scipy.stats import pearsonr
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import GridSearchCV




def model(p, L):
        # Read X-df and y_df from get_features.py 
        print("p:", p)
        print("L:", L)
        
        # Now get the labels
        # TEST SET
        # The PDBBind 2007 core set of 195 protein-ligand complexes is used as the test set.
        file = open("test_set_comps_final_for_CNN.txt", "r")
        data = file.read()
        test_set_comps = data.split('\n')

        file = open("v2007/INDEX.2007.core.data", "r")
        lines = file.readlines()
        test_set_labels = []
        # lines 1 - 5 are comments
        for x in lines[5:]:
        #print(x.split(' '))
                comp = x.split(' ')[0]
                if comp in test_set_comps:
                        if x.split(' ')[8] == "":
                                test_set_labels.append(x.split(' ')[7])
                        else:  
                                test_set_labels.append(x.split(' ')[8])
        file.close()
        #print(test_set_comps)
        print("number of elements in test set:",len(test_set_comps))
        print("number of elements in test set labels:",len(test_set_labels))
        test_labels_df = pd.DataFrame({'PDB_IDs': test_set_comps, 'bindingAffinity': test_set_labels})
        test_labels_df['bindingAffinity'] = test_labels_df['bindingAffinity'].astype(float)

        print(test_labels_df)
        # TRAINING SET
        # The PDBBind 2007 refined set, excluding the PDBBind 2007 core set, is used as the training set with 1105 protein-ligand complexes.
        file = open("train_set_comps_final_for_CNN.txt", "r")
        data = file.read()
        train_set_comps = data.split('\n')
        file = open("v2007/INDEX.2007.refined.data", "r")
        lines = file.readlines()
        train_set_labels = []
        for x in lines[5:]:
                comp = x.split(' ')[0]
                label = x.split(' ')[7]
                if (comp in train_set_comps) and (comp not in test_set_comps):
                        if label == "":
                                train_set_labels.append(x.split(' ')[6])
                        else:  
                                train_set_labels.append(label)
        file.close()
        print("number of elements in training set:",len(train_set_comps))
        print("number of elements in training set labels:",len(train_set_labels))

        train_labels_df = pd.DataFrame({'PDB_IDs': train_set_comps, 'bindingAffinity': train_set_labels})
        train_labels_df['bindingAffinity'] = train_labels_df['bindingAffinity'].astype(float)
        print(train_labels_df)
        
        X_train_df = pd.read_csv('X/X_train_electrostatic_p' + str(p) + '_L' + str(L) + '.csv', index_col=0)
        X_test_df = pd.read_csv('X/X_test_electrostatic_p' + str(p) + '_L' + str(L) + '.csv', index_col=0)
        print(X_train_df.head())
        print(X_test_df.head())
        
        y_train = np.array(train_labels_df.drop(['PDB_IDs'], axis=1))
        print("y_train", y_train[0:10])
        y_test = np.array(test_labels_df.drop(['PDB_IDs'], axis=1))
        print("y_test", y_test[0:10])

        train_data = pd.merge(X_train_df, train_labels_df, on='PDB_IDs' )
        print("train_data shape", train_data.shape)
        test_data = pd.merge(X_test_df, test_labels_df, on='PDB_IDs' )
        print("test_data shape", test_data.shape)

        full_df = pd.concat([train_data, test_data], axis = 0)
        full_df = full_df.drop(['PDB_IDs'], axis = 1)
        print("full_df shape:", full_df.shape)
        print(full_df)
        full_df = full_df.drop_duplicates()

        # Check for rows where all columns except the last one are NOT equal to 0
        mask = (full_df.iloc[:, :-1] != 0).any(axis=1)

        # Filter the DataFrame based on the condition
        full_df = full_df[mask]
        print("full_df shape after removing rows of all 0:", full_df.shape)

        full_df = full_df.drop(columns=full_df.columns[(full_df == 0).all(axis=0)])
        print("full_df shape after removing columns of all 0:", full_df.shape)

        # Remove outliers in based on interquartile range
        #lower_bound = full_df['bindingAffinity'].quantile(0.25)
        #upper_bound = full_df['bindingAffinity'].quantile(0.75)
       
        # Filter the DataFrame based on the bounds for the specified column
        #filtered_df = full_df[(full_df['bindingAffinity'] >= lower_bound) & (full_df['bindingAffinity'] <= upper_bound)]
        #full_df = filtered_df

        print("Shape after filtering out outliers:", full_df.shape)

        X = full_df.drop(['bindingAffinity'], axis=1)
        y = full_df['bindingAffinity']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

        scaler_X = MaxAbsScaler()

        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        print(X_train_scaled)
        #X_train_scaled = X_train
        #X_test_scaled = X_test

        #y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1, 1)).flatten()
        #y_test_scaled = scaler_y.transform(np.array(y_test).reshape(-1, 1)).flatten()

        #y_train_scaled = np.array(y_train)
        #y_test_scaled = np.array(y_test)

        # Create indices for the data points
        indices_train = np.arange(len(y_train))
        indices_test = np.arange(len(y_test))
        # Plot y_train and y_test data points as points
        plt.scatter(indices_train, y_train, label='y_train', color='blue')
        plt.scatter(indices_test, y_test, label='y_test', color='red')

        # Add labels and legend
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.legend()
        plt.savefig('y_output.png')

        # Initialize the model
        #rf = RandomForestRegressor(n_estimators=10, random_state=42)
        regressor = LinearRegression()
        # Train the model
        #rf.fit(X_train, y_train)
        regressor.fit(X_train_scaled, y_train)

        y_pred = regressor.predict(X_test_scaled)

        print(y_test)
        print(y_pred)
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred)
        plt.plot([min(y_test.min(), y_pred.min()), max(np.array(y_test).max(), y_pred.max())], [min(np.array(y_test).min(), y_pred.min()), max(np.array(y_test).max(), y_pred.max())], color='red')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('True vs Predicted Values for p = ' + str(p) + ' and L = ' + str(L))
        plt.savefig('plots_rf/scatter_p' + str(p) + '_L' + str(L) + '.png')
        
        
        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        pcc = np.corrcoef(np.array(y_test).squeeze(), np.array(y_pred).squeeze())[0, 1]

        # Specify the file path where you want to save the metrics
        file_path = 'evals_rf/eval_metrics_p' + str(p) + '_L' + str(L) + '.txt'
        # Open the file in write mode
        with open(file_path, "w") as file:
                # Write each metric along with its value to the file
                file.write("Evaluation Metrics for p = " + str(p) + ", L = " + str(L) + " : \n")
                file.write(f"Mean Squared Error (MSE): {mse}\n")
                file.write(f"Root Mean Squared Error (MSE): {rmse}\n")
                file.write(f"Mean Absolute Error (MAE): {mae}\n")
                file.write(f"Mean Absolute Percentage Error (MAE): {mape}\n")   
                file.write(f"R^2: {r2}\n")
                file.write(f"Pearson Correlation Coefficient: {pcc}\n")

     
        print("Metrics exported to:", file_path)






if __name__ == "__main__":
    # Extract command-line arguments
    p = int(sys.argv[1])
    L = int(sys.argv[2])

    model(p, L)   