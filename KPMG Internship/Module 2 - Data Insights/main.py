# %%

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import os
from joblib import load

# %% IMPORT DATA

def import_data():
    '''
    Returns the features (X_old) and labels (y) to train the classification model. Additionally, 
    it also returns the new features (X_new) to be feed to the trained model.

    Parameters
    ----------
    None

    Returns
    -------
    X_new, X_old, labels: tuple
        A tuple containing all numerical features for the ML model and the target to predict (Category)
    '''
     
    def load_old_customer_data():
        '''
        Returns the features (X) and labels (y) from the merged tabular data in a tuple. The features (X) only include numerical tabular data.

        Parameters
        ----------
        None

        Returns
        -------
        features, labels: tuple
            A tuple containing all the numerical features for the ML model and the target to predict (Category)
        '''
        
        current_directory = os.getcwd()
        csv_relative_directory = 'KPMG_VI_New_raw_data_update_final.xlsx'
        csv_directory = os.path.join(current_directory, csv_relative_directory)
        df = pd.read_excel(csv_directory, sheet_name="Merged Dataset (C+A+RFM)")
        labels = df['Customer Title']
        features = df[['gender','past_3_years_bike_related_purchases','Age Category','job_industry_category','wealth_segment','owns_car','tenure','state','property_valuation']]
        features['state'] = features['state'].str.replace('Vic','VIC')

        return features, labels

    def load_new_customer_data():
        '''
        Returns the features (X) from the new customers dataset in a tuple. The features (X) only includes numerical tabular data.

        Parameters
        ----------
        None

        Returns
        -------
        features: tuple
            A tuple containing all the numerical features for the ML model
        '''
        
        current_directory = os.getcwd()
        csv_relative_directory = 'KPMG_VI_New_raw_data_update_final.xlsx'
        csv_directory = os.path.join(current_directory, csv_relative_directory)
        df = pd.read_excel(csv_directory, sheet_name="NewCustomerList")
        features = df[['gender','past_3_years_bike_related_purchases','Age Category','job_industry_category','wealth_segment','owns_car','tenure','state','property_valuation']]
        features['state'] = features['state'].str.replace('Vic','VIC')
        
        return features

    X_old, y = load_old_customer_data()
    X_new = load_new_customer_data()

    return X_new, X_old, y

X_new, X_old, y = import_data()

# %% TRANSFORM & SPLIT DATA
'''  
gender - Label Encoding

past_3_years_bike_related_purchases - Divide into groups and Label encodin

Age Category - One hot encoding

job_industry_category - One hot encoding

wealth_segment - One Hot Encoding

owns_car - Label Encoding 0 and 1 

tenure - Same

State - One hot Encoding

Property Valuation - Same
'''
def transform_data(features_new, features_old, labels):

    def transform_labels_data(labels):
        # The label dictionary
        label_dict = {'Platinum': 3, 'Gold': 2, 'Silver': 1, 'Bronze': 0}
        # Use the map function to replace labels with encoded values
        labels = labels.map(label_dict)

        return labels

    def transform_features_data(features):

        def purchases_label_encoder(sales_column):
            # Calculate the quartiles (q1, median, q3)
            q1 = sales_column.quantile(0.25)
            median = sales_column.quantile(0.5)
            q3 = sales_column.quantile(0.75)
            # Define the bin edges for the label encoding
            bins = [sales_column.min(), q1, median, q3, sales_column.max()]
            # Define the bin labels
            labels = [0, 1, 2, 3]
            # Use pd.qcut to create the bins and map the values to the labels
            encoded_sales = pd.cut(sales_column, bins=bins, labels=labels, include_lowest=True)

            return encoded_sales

        def purchases_encoder(features):
            features['past_3_years_bike_related_purchases'] = purchases_label_encoder(features['past_3_years_bike_related_purchases'])

            return features


        def label_encoder(features, columns_to_label):
            #create instance of label encoder
            lab = LabelEncoder()
            #perform label encoding on desired column
            for column in columns_to_label:
                features[column] = lab.fit_transform(features[column])

            return features


        def one_hot_encoder(features, columns_to_label):
            # Create an instance of the OneHotEncoder
            encoder = OneHotEncoder(handle_unknown='ignore')
            # Fit and transform the selected columns
            encoded_features = encoder.fit_transform(features[columns_to_label])
            # Create a DataFrame from the encoded features
            encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names(columns_to_label)).astype('int64')
            # Drop the original columns from the DataFrame
            features = features.drop(columns_to_label, axis=1)
            # Concatenate the original DataFrame with the encoded DataFrame
            features = pd.concat([features, encoded_df], axis=1)

            return features

        def encoder(features):
            # One hot encoding
            features = one_hot_encoder(features,['Age Category', 'job_industry_category', 'wealth_segment', 'state'])
            # Label encoding
            features = label_encoder(features,['gender','past_3_years_bike_related_purchases','owns_car'])

            return features

        features = purchases_encoder(features)
        features = encoder(features)

        return features

    X_old = transform_features_data(features_old)
    X_new = transform_features_data(features_new)
    y = transform_labels_data(labels)

    return X_new, X_old, y

X_new, X_old, y = transform_data(X_new, X_old, y)

# %% TRAIN THE MODEL

# Models and their hyperparameter grids for tuning
models = {
    'DecisionTree': (DecisionTreeClassifier(), {'max_depth': [None, 5, 10, 15], 'criterion': ['gini', 'entropy', 'log_loss'],'max_features': [1.0,'sqrt', 'log2', None]}),
    'RandomForestClassifier': (RandomForestClassifier(),{'max_depth': [None, 5, 10, 15], 'criterion': ['gini', 'entropy', 'log_loss'],'max_features': [1.0,'sqrt', 'log2', None]}),
    'SVM': (SVC(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
    'KNN': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
    'NaiveBayes': (GaussianNB(), {}),
    'GradientBoostingClassifier': (GradientBoostingClassifier(),{'max_depth': [None,3, 5, 10, 15], 'criterion': ['friedman_mse', 'squared_error'],'learning_rate': [0.1, 0.5, 1],'n_estimators': [10, 50, 70, 100]}),
}

def cross_validate_and_tune(model, params, X, y, n_splits=5):
    # Initialize StratifiedKFold for cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(model, params, cv=cv, n_jobs=-1)

    # Perform cross-validation and hyperparameter tuning
    grid_search.fit(X, y)

    return grid_search.best_params_, grid_search.best_score_

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_old, y, test_size=0.2, random_state=42)

def classification_matrix(labels, predictions, clf):
    '''
        Creates a normalised confusion matrix for a model, its labels and its predictions

        Parameters
        ----------
        labels: pandas.core.series.Series
            A pandas DataFrame containing the targets/labels 

        predictions: pandas.core.series.Series
            A pandas series containing the predictions of the model

        clf: sklean.model
            An instance of the sklearn classifier model

        Returns
        -------
        None
    '''

    cm = confusion_matrix(labels, predictions, normalize=None, labels=clf.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    display.plot()
    plt.show()

    return

for model_name, (model, params) in models.items():
    best_params, best_score = cross_validate_and_tune(model, params, X_train, y_train)
    print(f"Best parameters for {model_name}: {best_params}")
    print(f"Best cross-validation score for {model_name}: {best_score:.4f}")

    # Train the model with best parameters on the full training set
    model.set_params(**best_params)
    model.fit(X_train, y_train)
    y_pred_test= model.predict(X_test)

    best_metrics = {
        "F1 score" : f1_score(y_test, y_pred_test, average="macro"),
        "Precision":  precision_score(y_test, y_pred_test, average="macro"),
        "Recall" :  recall_score(y_test, y_pred_test, average="macro"),
        "Accuracy" :  accuracy_score(y_test, y_pred_test)
    }

    classification_matrix(y_test, y_pred_test,model)
    print(best_metrics)

    # Evaluate the model on the test set
    test_score = model.score(X_test, y_test)
    print(f"Test score for {model_name}: {test_score:.4f}")
    print()




# %%
