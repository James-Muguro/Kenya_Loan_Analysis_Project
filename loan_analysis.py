# To start working with the dataset, we will import the following libraries

import pandas as pd  # Pandas for data manipulation and analysis
import numpy as np   # NumPy for numerical operations
import matplotlib.pyplot as plt  # Matplotlib for basic data visualization
import seaborn as sns  # Seaborn for advanced data visualization
from sklearn.cluster import KMeans  # KMeans for clustering
from sklearn.preprocessing import StandardScaler  # StandardScaler for feature scaling
from sklearn.model_selection import train_test_split  # Train-test split for model evaluation
from sklearn.linear_model import LogisticRegression  # Logistic Regression for classification
from sklearn.metrics import f1_score, confusion_matrix  # Metrics for model evaluation
from sklearn.svm import SVC  # Support Vector Classifier for classification
import warnings  # Warnings to suppress unnecessary warnings
warnings.filterwarnings("ignore")


# Loading the data
import ipywidgets as widgets
from IPython.display import display, clear_output
from tkinter import Tk, filedialog
import pandas as pd

def select_file(b):
    clear_output()
    root = Tk()
    root.withdraw()  # Hide the main window
    root.call('wm', 'attributes', '.', '-topmost', True)  # Raise the root to the top of all windows
    b.files = filedialog.askopenfilename(multiple=False)  # List of selected files
    path = b.files
    global df
    df = pd.read_excel(path)
    print(f'Loaded dataframe from {path}')

fileselect = widgets.Button(description="File select")
fileselect.on_click(select_file)

display(fileselect)
 
display(df.head())


# Let's visualize the categorical and ordinal features.

# The Dataset containing the categorical columns
parameters = ['Loan_Status', 'Married', 'Self_Employed', 'age','Credit_History']

# Create subplots for each parameter
fig, axes = plt.subplots(nrows=len(parameters), ncols=1, figsize=(6, 4 * len(parameters)))

# Specify the bar width
bar_width = 0.4

for i, param in enumerate(parameters):
    sns.countplot(x=param, data=df, palette='viridis', ax=axes[i], width=bar_width)
    axes[i].set_title(f'Distribution of {param}')
    axes[i].set_xlabel(param)
    axes[i].set_ylabel('Count')

plt.tight_layout()
plt.show()


# Now let's visualize the ordinal variables.

# Plotting 'Dependents'
df['Dependents'].value_counts(normalize=True).plot.bar(title="Dependents")
plt.show()

# Plotting 'Education'
df['Education'].value_counts(normalize=True).plot.bar(title="Education")
plt.show()

# Plotting 'Property_Area'
df['Property_Area'].value_counts(normalize=True).plot.bar(title="Property Area")
plt.show()

# Plotting 'county'
df['county'].value_counts(normalize=True).plot.bar(title="County")
plt.show()



# Finally lets visualize the numerical variables

# Set up subplots
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))

# Plot for ApplicantIncome
sns.distplot(df['ApplicantIncome'], hist=False, rug=True, label='ApplicantIncome', ax=axes[0])
axes[0].set_title('Distribution of ApplicantIncome')

# Plot for CoapplicantIncome
sns.distplot(df['CoapplicantIncome'], hist=False, rug=True, label='CoapplicantIncome', ax=axes[1])
axes[1].set_title('Distribution of CoapplicantIncome')

# Plot for LoanAmount
sns.distplot(df['LoanAmount'], hist=False, rug=True, label='LoanAmount', ax=axes[2])
axes[2].set_title('Distribution of LoanAmount')

# Adjust layout
plt.tight_layout()

# Display the plots
plt.show()



# Cleaning the Data
# Missing Values in the dataset
missing_values = df.isnull().sum()
print(missing_values)

# Since the dataset seems relatively small (615 observations), dropping rows can lead to a substantial loss of information.
# Hence we will fill missing values in categorical columns with the mode (most frequent value) 
# and in numerical columns with the median


# Define columns with missing values
columns_with_missing = ['Gender', 'Married', 'Dependents', 'Self_Employed',
                         'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

# Impute missing values for categorical columns with the mode
for col in columns_with_missing:
    if df[col].dtype == 'O':  # 'O' indicates object data type (categorical)
        df[col].fillna(df[col].mode()[0], inplace=True)

# Impute missing values for numerical columns with the median
for col in columns_with_missing:
    if df[col].dtype != 'O':  # Not categorical (numerical)
        df[col].fillna(df[col].median(), inplace=True)

# Display the cleaned DataFrame
df.head(10)


# Next we Convert the 'application_date' column to a consistent date format.

# Convert 'application_date' column to datetime format with dayfirst=True
df['application_date'] = pd.to_datetime(df['application_date'], dayfirst=True, errors='coerce')


# Display the DataFrame to verify the changes
df.head(10)


# Analyzing the Data
# Summary statistics.

df.drop('application_date', axis=1).describe()


# To understand the distribution of loans across counties and user demographics, 
# we will generate the below statistics and visualizations.

# Summary statistics for loan distribution across counties
county_summary = df.groupby('county')['Loan_Status'].value_counts().unstack().fillna(0)
print("Loan Distribution Across Counties:")
print(county_summary)

# Summary statistics for loan distribution based on user demographics (e.g., Gender and Education)
gender_summary = df.groupby('Gender')['Loan_Status'].value_counts().unstack().fillna(0)
education_summary = df.groupby('Education')['Loan_Status'].value_counts().unstack().fillna(0)

print("\nLoan Distribution by Gender:")
print(gender_summary)

print("\nLoan Distribution by Education:")
print(education_summary)


# The below visualizations will give insights into the distribution of loans
# and the approval status across different categories

# Visualize loan distribution across counties
plt.figure(figsize=(12, 6))
sns.countplot(x='county', data=df, hue='Loan_Status')
plt.title('Loan Distribution Across Counties')
plt.xlabel('County')
plt.ylabel('Loan Count')
plt.show()

# Visualize loan distribution based on user demographics (e.g., Gender and Education)
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='Gender', data=df, hue='Loan_Status')
plt.title('Loan Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Loan Count')

plt.subplot(1, 2, 2)
sns.countplot(x='Education', data=df, hue='Loan_Status')
plt.title('Loan Distribution by Education')
plt.xlabel('Education')
plt.ylabel('Loan Count')

plt.tight_layout()
plt.show()


# The below generates a pair plot using the Seaborn library, visualizing the relationships between different 
# pairs of numerical variables in the dataset

# Set the style
sns.set(style="whitegrid")

# Create a pair plot
plt.figure(figsize=(12, 8))

# Adjusting the layout before creating the pair plot
plt.tight_layout()

sns.pairplot(df, hue="Loan_Status", height=3)

# Show the plot
plt.show()


# This code will generate a set of count plots for each categorical variable, showcasing the distribution of 
# loan statuses (Y or N) among different categories.

# Set the style
sns.set(style="whitegrid")

# Create subplots for each categorical variable
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))

# Categorical variables to analyze
categorical_variables = ['Gender', 'Married', 'Dependents', 'county', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']

# Plotting
for i, variable in enumerate(categorical_variables):
    sns.countplot(x=variable, data=df, hue='Loan_Status', ax=axes[i // 4, i % 4])
    axes[i // 4, i % 4].set_title(f'Distribution of {variable}')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# Extract month and year from 'application_date'
df['YearMonth'] = df['application_date'].dt.to_period('M')

# Group by 'YearMonth' and count the number of loans
loan_evolution = df.groupby('YearMonth').size()

# Plotting
plt.figure(figsize=(12, 6))
loan_evolution.plot(marker='o')
plt.title('Evolution of the Loan Book Over Time')
plt.xlabel('Year-Month')
plt.ylabel('Number of Loans')
plt.grid(True)
plt.show()


# Here we will perform K-Means clustering on 'ApplicantIncome' and 'LoanAmount', using the Elbow method to determine
# the optimal number of clusters, and then visualizes the distinct groups of borrowers based on these features


# 'ApplicantIncome' and 'LoanAmount' are our relevant features for clustering
features_for_clustering = ['ApplicantIncome', 'LoanAmount']

# Select relevant columns
selected_features = df[features_for_clustering]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_features)

# Use the Elbow method to determine the optimal number of clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the Elbow method
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()

# Based on the Elbow method, choose the optimal number of clusters and fit the model
optimal_k = 3  # Choose the number based on the Elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualize the clusters
plt.figure(figsize=(10, 6))
for cluster in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['ApplicantIncome'], cluster_data['LoanAmount'], label=f'Cluster {cluster}')

plt.title('Distinct Groups of Borrowers')
plt.xlabel('Applicant Income')
plt.ylabel('Loan Amount')
plt.legend()
plt.show()


#Mapping columns containing strings
mapping = {
    'Gender': {'Male': 0, 'Female': 1},
    'Married': {'Yes': 1, 'No': 0},
    'Education': {'Graduate': 1, 'Not Graduate': 0},
    'Self_Employed': {'Yes': 1, 'No': 0},
    'Property_Area': {'Semiurban': 1, 'Urban': 2, 'Rural': 3},
    'Loan_Status': {'Y': 1, 'N': 0},
    'county': {'Nairobi': 1, 'Kiambu': 2, 'Machakos': 3, 'Mombasa':4},
    'Dependents': {0: 0, 1: 1, 2: 2, '3+': 3}
}

df.replace(mapping, inplace=True)


# Drop the 'Loan_ID' column
df.drop('Loan_ID', axis=1, inplace=True)

# Convert 'application_date' to a numeric representation (e.g., number of days since January 1st, 2017)
df['application_date'] = (df['application_date'] - pd.to_datetime('2017-01-01')).dt.days

# Drop the 'YearMonth' column
df.drop('YearMonth', axis=1, inplace=True)


# Split the data into training and testing sets
X_1=df.drop('Loan_Status',axis=1)
y_1=df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X_1, y_1, test_size=0.20, random_state=42)


# Define class weights as a dictionary
class_weights = {0: 0.7, 1: 0.3}

# Create a logistic regression model with class weights
lr = LogisticRegression(class_weight=class_weights, max_iter=100)

# Fit the model and make predictions
pred = lr.fit(X_train, y_train).predict(X_test)

# Calculate F1 score
f1 = f1_score(y_test, pred)

# Print the F1 score
print("F1 Score:", f1)



# calculating the confusion matrix using scikit-learn

cm = confusion_matrix(pred, y_test)
cm


#  The actual outcomes (labels) corresponding to the input data points that were set aside for testing the 
# model's performance

y_test


# A heatmap showing the patterns of association between different variables in the dataset

sns.heatmap(df.corr())


# 1. Loan approval rate:

# Calculate overall loan approval rate
total_applications = len(df)
approved_applications = len(df[df['Loan_Status'] == 1])  # Using mapped value for 'Y'
approval_rate_overall = (approved_applications / total_applications) * 100

print(f"Overall Loan Approval Rate: {approval_rate_overall:.2f}%")

# Check unique values in 'Loan_Status' column
unique_loan_status_values = df['Loan_Status'].unique()
print(f"Unique values in 'Loan_Status' column: {unique_loan_status_values}")

# Calculate loan approval rate by Gender
if 1 in unique_loan_status_values:
    approval_rate_by_gender = df.groupby('Gender')['Loan_Status'].value_counts(normalize=True).loc[:, 1] * 100  # Using mapped value for 'Y'
    print("\nLoan Approval Rate by Gender:")
    print(approval_rate_by_gender)
else:
    print("\nMapped value '1' not found in 'Loan_Status' column for loan approval rate by Gender.")



# 2. Debt-to-income ratio:

df['Debt_Income_Ratio'] = df['LoanAmount'] / (df['ApplicantIncome'] + df['CoapplicantIncome'])

# Display the updated DataFrame
print(df[['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome', 'Debt_Income_Ratio']])


# 3. Credit ranking:

# Create a new column for credit history categories
df['Credit_History_Category'] = df['Credit_History'].map({1: 'Good', 0: 'Bad'})

# Analyze differences in loan terms, age groups, and approval rates by credit history category
credit_history_analysis = df.groupby('Credit_History_Category').agg({
    'Loan_Amount_Term': 'mean',  # Mean loan term
    'age': ['mean', 'median'],    # Mean and median age
    'Loan_Status': lambda x: x.value_counts(normalize=True).get(1, 0) * 100  # Approval rate using mapped value for 'Y'
}).reset_index()

# Display the analysis results
print(credit_history_analysis)



# Income groups:

# Create income brackets
income_bins = [0, 5000, 10000, 15000, np.inf]
income_labels = ['0-5000', '5001-10000', '10001-15000', '15001+']

# Create a new column for income brackets
df['Income_Bracket'] = pd.cut(df['ApplicantIncome'], bins=income_bins, labels=income_labels, right=False)

# Analyze loan approval rates, loan amounts, and other variables across income brackets
income_analysis = df.groupby('Income_Bracket').agg({
    'Loan_Status': lambda x: x.value_counts(normalize=True).get(1, 0) * 100,  # Approval rate
    'LoanAmount': 'mean',  # Mean loan amount
    'CoapplicantIncome': 'mean',  # Mean coapplicant income
    # Add more variables as needed for analysis
}).reset_index()

# Display the analysis results
print(income_analysis)



# Age groups:

# Create age brackets
age_bins = [18, 25, 35, 45]
age_labels = ['18-25', '26-35', '36-45']

# Create a new column for age brackets
df['Age_Bracket'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

# Analyze loan approval rates, loan amounts, and marital status across age brackets
age_analysis = df.groupby('Age_Bracket').agg({
    'Loan_Status': lambda x: x.value_counts(normalize=True).get(1, 0) * 100,  # Approval rate
    'LoanAmount': 'mean',  # Mean loan amount
    'Married': lambda x: x.value_counts(normalize=True).get(1, 0) * 100,  # Percentage of married applicants
    # Add more variables as needed for analysis
}).reset_index()

# Display the analysis results
print(age_analysis)


# Overconfidence bias:

# Scatter plot: Income vs Loan Amount
plt.figure(figsize=(12, 6))
sns.scatterplot(x='ApplicantIncome', y='LoanAmount', hue='Loan_Status', data=df)
plt.title('Income vs Loan Amount')
plt.xlabel('Applicant Income')
plt.ylabel('Loan Amount')
plt.show()

# Box plot: Income distribution by Loan Status
plt.figure(figsize=(10, 6))
sns.boxplot(x='Loan_Status', y='ApplicantIncome', data=df)
plt.title('Income Distribution by Loan Status')
plt.xlabel('Loan Status')
plt.ylabel('Applicant Income')
plt.show()

# Approval rate by income bracket
income_bins = [0, 5000, 10000, 15000, 20000, float('inf')]
income_labels = ['0-5000', '5001-10000', '10001-15000', '15001-20000', '20001+']
df['Income_Bracket'] = pd.cut(df['ApplicantIncome'], bins=income_bins, labels=income_labels, right=False)

approval_by_income = df.groupby('Income_Bracket')['Loan_Status'].value_counts(normalize=True).unstack()
approval_by_income[1].plot(kind='bar', figsize=(12, 6), title='Loan Approval Rate by Income Bracket')
plt.xlabel('Income Bracket')
plt.ylabel('Approval Rate')
plt.show()



