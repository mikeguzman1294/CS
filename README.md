# URL Classification Challenge

## Challenge & Solution Debrief
***

In this challenge the goal is to solve a **multiclass classification** problem. The training dataset consists of web page URLs with associated page category labels.

Engineered features are made up of 2 main groups:

1. URL section parameters count.
2. URL section vectorized parameters using a bag of words approach and a TF-IDF transformation.

The data transformation pipeline also uses general data preprocessing approaches such as deletion of duplicated instances and handling missing values (completion upon a URL column information in the case of missing prefixes and deletion otherwise).

The ML algortihm section aims to iterate on a single type of predictor due to development time constraints. The chosen predictor is a Decision Tree model as base learner in addition to a bagging ensemble method in the form of a Random Forest classifier.

Techniques for handling unbalanced datasets such as stratified train/test split and class weighting are evaluated.

The best performant model is a Random Forest Classifier with 120 estimators and gini impurity criterion. Class weighting does not seem to improve the model performance. Its **macro average F1-score** is of **81%**. Its accuracy is of 94.7%, however it is important to consider that accuracy is a extremely unrealible metric in this case where the dependent variable classes are very unbalanced.

*Further work* on this classification approach may include more advanced imbalanced classes handling techniques, such as undersampling, oversampling (SMOTE) or generation of synthetic samples and testing more models, such as ensemble methods with other base learners and neural network classifiers along with more robust and extensive hyperparameter tuning processes. Also the evaluation of lemmatization and the usage of user defined vocabularies for the bag-of-words rather than using term frequency could be explored for the text processing.

The solution is implemented, commented and discussed in a Jupyter Notebook under the file name `URL_Prediction.ipynb`.

The notebook is divided in 6 main sections.

1.   **Environment Setup**
2.   **Exploratory Data Analysis**
3.   **Data Preprocessing**
4.   **Feature Engineering**
5.   **Classification Modelling**
6.   **Concluding Remarks**

A deep explanation of all the solution stages can be found in the aforementioned Jupyter notebook.

## Environment Setup
***

*This tutorial assumes that a distribution of Python 3.7+ is installed in a local computer. If not installed, download the latest version through this link <https://www.python.org/downloads/>*

*The set of command line instructions is intended to run in a Windows machine*

IMPORTANT: All the files contained in the zip file MUST remain in the same directory after decompressing.
***
### **1. Create and Launch Python Virtual Environment**

**1.1**

In order to manage the required Python libraries for this project it is **recommended** to create a Python virtual environment in the project's root folder.

To create a virtual environment use the following command:

```
python -m venv venv
```
*Note: The sencond venv in this command is the name the created virtual environment will receive. It is recommended to use 'venv' by convention.*

**1.2**

To activate the venv run the following command:
```
venv\scripts\activate
```
**1.3**

To install the required libraries for the challenge run the following command:
```
pip install -r requirements.txt
```
This step should only be done once after creating the virtual environment!

**Note: A requirements.yml file si also provided in case the user wants to reproduce the experiments in an Anaconda Virtual Environment.*

### **2. Run the Jupyter Notebook**

**2.1**

Start the notebook server from the command line:

```
jupyter notebook
```

This will print some information about the notebook server in your terminal, including the URL of the web application (by default, http://localhost:8888):

```
$ jupyter notebook
[I 08:58:24.417 NotebookApp] Serving notebooks from local directory: /Users/catherine
[I 08:58:24.417 NotebookApp] 0 active kernels
[I 08:58:24.417 NotebookApp] The Jupyter Notebook is running at: http://localhost:8888/
[I 08:58:24.417 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
```

It will then open your default web browser to this URL.

When the notebook opens in your browser, you will see the Notebook Dashboard, which will show a list of the notebooks, files, and subdirectories in the directory where the notebook server was started.

**2.2**

In the Dashboard, select the Jupyter Notebook named `URL_Prediction.ipynb`. This will open the notebook in a new tab of your explorer.

In the "Cell" tab of the notebook, select the option "Run All". This will run all the notebook cells and reproduce all the experiments and results with no additional effort.

**2.3**

Enjoy :)