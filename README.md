# Deep-Learning-Charity-Funding-Predictor

## Background

The non-profit foundation Alphabet Soup wants to create an algorithm to predict whether or not applicants for funding will be successful. With my knowledge of machine learning and neural networks, I'll use the features in the provided dataset to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, I have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

* **EIN** and **NAME**—Identification columns
* **APPLICATION_TYPE**—Alphabet Soup application type
* **AFFILIATION**—Affiliated sector of industry
* **CLASSIFICATION**—Government organization classification
* **USE_CASE**—Use case for funding
* **ORGANIZATION**—Organization type
* **STATUS**—Active status
* **INCOME_AMT**—Income classification
* **SPECIAL_CONSIDERATIONS**—Special consideration for application
* **ASK_AMT**—Funding amount requested
* **IS_SUCCESSFUL**—Was the money used effectively

## Instructions

### Step 1: Preprocess the data

Using my knowledge of Pandas and the Scikit-Learn’s `StandardScaler()`, I'll need to preprocess the dataset in order to compile, train, and evaluate the neural network model later in Step 2

Using the information we have provided in the starter code, follow the instructions to complete the preprocessing steps.

1. Read in the charity_data.csv to a Pandas DataFrame, and made sure to identify the following in my dataset:
  * What variable(s) are considered the target(s) for my model?
  * What variable(s) are considered the feature(s) for my model?
2. Dropped the `EIN` and `NAME` columns.
3. Determined the number of unique values for each column.
4. For those columns that have more than 10 unique values, determined the number of data points for each unique value.
6. Used the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then checked if the binning was successful.
7. Used `pd.get_dummies()` to encode categorical variables

### Step 2: Compile, Train, and Evaluate the Model

Using my knowledge of TensorFlow, I'll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. I'll need to think about how many inputs there are before determining the number of neurons and layers in my model. Once I've completed that step, I'll compile, train, and evaluate my binary classification model to calculate the model’s loss and accuracy.

1. Continued using the jupter notebook where I've already performed the preprocessing steps from Step 1.
2. Created a neural network model by assigning the number of input features and nodes for each layer using Tensorflow Keras.
3. Created the first hidden layer and choose an appropriate activation function.
4. If necessary, added a second hidden layer with an appropriate activation function.
5. Created an output layer with an appropriate activation function.
6. Checked the structure of the model.
7. Compiled and train the model.
8. Created a callback that saves the model's weights every 5 epochs.
9. Evaluated the model using the test data to determine the loss and accuracy.
10. Saved and exported my results to an HDF5 file, and name it `AlphabetSoupCharity.h5`.

### Step 3: Optimize the Model

Using my knowledge of TensorFlow, optimized my model in order to achieve a target predictive accuracy higher than 75%. If I couldn't achieve an accuracy higher than 75%, I made at least three attempts to do so.

Optimized my model in order to achieve a target predictive accuracy higher than 75% by using any or all of the following:

* Adjusting the input data to ensure that there are no variables or outliers that are causing confusion in the model, such as:
  * Dropping more or fewer columns.
  * Creating more bins for rare occurrences in columns.
  * Increasing or decreasing the number of values for each bin.
* Adding more neurons to a hidden layer.
* Adding more hidden layers.
* Using different activation functions for the hidden layers.
* Adding or reducing the number of epochs to the training regimen.

**NOTE**: I will not lose points if your model does not achieve target performance, as long as you make three attempts at optimizing the model in your jupyter notebook.

1. Created a new Jupyter Notebook file and name it `AlphabetSoupCharity_Optimzation.ipynb`.
2. Imported my dependencies, and read in the `charity_data.csv` to a Pandas DataFrame.
3. Preprocessed the dataset like I did in Step 1, taking into account any modifications to optimize the model.
4. Designed a neural network model, taking into account any modifications that will optimize the model to achieve higher than 75% accuracy.
5. Saved and exported my results to an HDF5 file, and name it `AlphabetSoupCharity_Optimization.h5`.

### Step 4: Write a Report on the Neural Network Model

For this part of the Challenge, I'll write a report on the performance of the deep learning model I created for AlphabetSoup.

The report should contain the following:

1. **Overview** of the analysis: Explain the purpose of this analysis.

2. **Results**: Using bulleted lists and images to support my answers, address the following questions.

  * Data Preprocessing
    * What variable(s) are considered the target(s) for my model?
    * What variable(s) are considered to be the features for my model?
    * What variable(s) are neither targets nor features, and should be removed from the input data?
  * Compiling, Training, and Evaluating the Model
    * How many neurons, layers, and activation functions did I select for my neural network model, and why?
    * Was I able to achieve the target model performance?
    * What steps did I take to try and increase model performance?

3. **Summary**: Summarized the overall results of the deep learning model. Included a recommendation for how a different model could solve this classification problem, and explained my recommendation.
