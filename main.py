import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
import warnings

warnings.filterwarnings("ignore")


def load_data(file_path):
    """
    Loads the loan data from an Excel file.

    Args:
        file_path (str): The path to the Excel file.

    Returns:
        pandas.DataFrame: The loaded dataframe.
    """
    try:
        df = pd.read_excel(file_path)
        print(f"Loaded dataframe from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None


def clean_data(df):
    """
    Cleans and preprocesses the loan data.

    Args:
        df (pandas.DataFrame): The input dataframe.

    Returns:
        pandas.DataFrame: The cleaned dataframe.
    """
    # Impute missing values
    columns_with_missing = ['Gender', 'Married', 'Dependents', 'Self_Employed',
                            'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    for col in columns_with_missing:
        if df[col].dtype == 'O':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    # Convert 'application_date' to datetime
    df['application_date'] = pd.to_datetime(df['application_date'], dayfirst=True, errors='coerce')

    return df

def plot_categorical_features(df):
    """
    Visualizes the distribution of categorical features.

    Args:
        df (pandas.DataFrame): The input dataframe.
    """
    parameters = ['Loan_Status', 'Married', 'Self_Employed', 'age', 'Credit_History']
    fig, axes = plt.subplots(nrows=len(parameters), ncols=1, figsize=(6, 4 * len(parameters)))
    bar_width = 0.4
    for i, param in enumerate(parameters):
        sns.countplot(x=param, data=df, palette='viridis', ax=axes[i], width=bar_width)
        axes[i].set_title(f'Distribution of {param}')
        axes[i].set_xlabel(param)
        axes[i].set_ylabel('Count')
    plt.tight_layout()
    plt.show()


def plot_ordinal_features(df):
    """
    Visualizes the distribution of ordinal features.

    Args:
        df (pandas.DataFrame): The input dataframe.
    """
    df['Dependents'].value_counts(normalize=True).plot.bar(title="Dependents")
    plt.show()
    df['Education'].value_counts(normalize=True).plot.bar(title="Education")
    plt.show()
    df['Property_Area'].value_counts(normalize=True).plot.bar(title="Property Area")
    plt.show()
    df['county'].value_counts(normalize=True).plot.bar(title="County")
    plt.show()


def plot_numerical_features(df):
    """
    Visualizes the distribution of numerical features.

    Args:
        df (pandas.DataFrame): The input dataframe.
    """
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))
    sns.distplot(df['ApplicantIncome'], hist=False, rug=True, label='ApplicantIncome', ax=axes[0])
    axes[0].set_title('Distribution of ApplicantIncome')
    sns.distplot(df['CoapplicantIncome'], hist=False, rug=True, label='CoapplicantIncome', ax=axes[1])
    axes[1].set_title('Distribution of CoapplicantIncome')
    sns.distplot(df['LoanAmount'], hist=False, rug=True, label='LoanAmount', ax=axes[2])
    axes[2].set_title('Distribution of LoanAmount')
    plt.tight_layout()
    plt.show()

def perform_clustering(df):
    """
    Performs K-Means clustering on borrower data.

    Args:
        df (pandas.DataFrame): The input dataframe.

    Returns:
        pandas.DataFrame: The dataframe with cluster labels.
    """
    features_for_clustering = ['ApplicantIncome', 'LoanAmount']
    selected_features = df[features_for_clustering]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(selected_features)

    # Elbow method to find optimal K
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)

    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.show()

    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    plt.figure(figsize=(10, 6))
    for cluster in range(optimal_k):
        cluster_data = df[df['Cluster'] == cluster]
        plt.scatter(cluster_data['ApplicantIncome'], cluster_data['LoanAmount'], label=f'Cluster {cluster}')

    plt.title('Distinct Groups of Borrowers')
    plt.xlabel('Applicant Income')
    plt.ylabel('Loan Amount')
    plt.legend()
    plt.show()

    return df

def train_and_evaluate_model(df):
    """
    Trains a logistic regression model and evaluates its performance.

    Args:
        df (pandas.DataFrame): The input dataframe.
    """
    mapping = {
        'Gender': {'Male': 0, 'Female': 1},
        'Married': {'Yes': 1, 'No': 0},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Property_Area': {'Semiurban': 1, 'Urban': 2, 'Rural': 3},
        'Loan_Status': {'Y': 1, 'N': 0},
        'county': {'Nairobi': 1, 'Kiambu': 2, 'Machakos': 3, 'Mombasa': 4},
        'Dependents': {0: 0, 1: 1, 2: 2, '3+': 3}
    }
    df.replace(mapping, inplace=True)

    df.drop('Loan_ID', axis=1, inplace=True)
    df['application_date'] = (df['application_date'] - pd.to_datetime('2017-01-01')).dt.days

    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    class_weights = {0: 0.7, 1: 0.3}
    lr = LogisticRegression(class_weight=class_weights, max_iter=100)

    pred = lr.fit(X_train, y_train).predict(X_test)

    f1 = f1_score(y_test, pred)
    print(f"F1 Score: {f1}")

    cm = confusion_matrix(pred, y_test)
    print(f"Confusion Matrix:\n{cm}")

    # ROC Curve
    y_pred_proba = lr.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def main():
    """
    Main function to run the loan analysis.
    """
    # Load data
    file_path = "/home/james-kamande/Documents/Projects/Kenya_Loan_Analysis_Project/loan_data.xlsx"
    df = load_data(file_path)
    if df is None:
        return

    # Clean data
    df = clean_data(df.copy())

    # Perform EDA
    plot_categorical_features(df.copy())
    plot_ordinal_features(df.copy())
    plot_numerical_features(df.copy())

    # Perform clustering
    df_clustered = perform_clustering(df.copy())

    # Train and evaluate model
    train_and_evaluate_model(df_clustered.copy())


if __name__ == "__main__":
    main()
