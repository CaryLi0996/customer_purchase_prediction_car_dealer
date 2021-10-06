Predicting Customers' Purchases using Logistic Regression: Project Overview
============================================================================
* Objective: This project was created to help company, XYZ car dealer, to discover potential clients based on existing customers' data. The company’s analytic department concluded that a logistic regression model should be adopted to to predict whether the customer buys the car or not. By building a logistic regression model, the marketing team will be able to extract insightful information based on each of our customers’ purchasing behaviors and other characteristics. This helps segment the potential customers in our contact list into defined groups and predict which customers are more likely to make a purchase. 


Datasets Overview 
----------------------------
The original customer dataset contains 502 rows of customer data, with each row representing a unique customer. The purchase column was categorized into ‘0’ which refers to the customers who didn’t purchase a car and ‘1’ for those who made a purchase.



Data Exploration and Preprocessing
----------------------------------
After collecting the dataset and importing it as a dataframe, I did the following exploration and cleaning: 

1. keeping only four variables that are important in classifying our customer groups, which are the sex, age, income, and purchase columns.
2. check key data statistics 
3. drop missing values 
4. convert categorical values into dummy variables 
5. check correlations between each of the variables 

[![Screen-Shot-2021-10-06-at-6-20-44-PM.png](https://i.postimg.cc/zvzPDPLw/Screen-Shot-2021-10-06-at-6-20-44-PM.png)](https://postimg.cc/ZBQLVHx0)


Model Building & Prediction
----------------------------------
A logistic regression model would be applied in this analysis as it works particularly well in binary classification tasks, with a probability output generated to predict the likelihood of an input belonging to the category output of ‘1’, based on all its features. Moreover, from the correlation table above, one can see that there is little to no multicollinearity among the independent variables, which fits the requirement of logistic regression.

It is also important to find out which single or combination of features has the greatest effect on the purchase decisions of the customers. An iteration tool can be applied on the logistic regression algorithm, where it constructs a unique model with different parameters when iterating through all the single and combination of features each time. 

7 different logisitc regression models are built in this case: 

[![Screen-Shot-2021-10-06-at-6-28-11-PM.png](https://i.postimg.cc/JnNzCtJD/Screen-Shot-2021-10-06-at-6-28-11-PM.png)](https://postimg.cc/WF4vMN5T)

[![Screen-Shot-2021-10-06-at-6-29-03-PM.png](https://i.postimg.cc/7L7xSbKs/Screen-Shot-2021-10-06-at-6-29-03-PM.png)](https://postimg.cc/jWx0bq0P)

The prediction results are listed in the train and test columns in the array form, where 0 indicates ‘not purchase’ and 1 refers to ‘purchase’.

[![Screen-Shot-2021-10-06-at-6-31-33-PM.png](https://i.postimg.cc/rsHbhNmS/Screen-Shot-2021-10-06-at-6-31-33-PM.png)](https://postimg.cc/9wGJ0TyM)


Model Evaluation
-------------------
By evaluating each model’s prediction against the known true class values in the form of a confusion matrix and classification report, one can see that the logistic regression model based on sex and age features has the highest accuracy and F-1 scores in the models’ prediction. This could infer that among the 3 features (sex, age, income), customers’ (income, age) have the largest effect on their purchase decision. 

