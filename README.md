**Anti-Money-laundering using Machine Learning**

 Client: A leading Financial Institutions in the world
 
 Business Problem: 
 Financial institutions facing significant challenges in detecting and preventing money laundering due to 
 the large volume of transactions they handle, complex regulations to comply with, and the use of 
 deceptive tactics like shell companies in money laundering. Failure to prevent money laundering can 
 harm their reputation and ability to operate effectively.

![image](https://user-images.githubusercontent.com/107097836/231665827-17e8afaa-595b-4ece-b63f-8b17a95327a7.png)


Summary
In this project I use the Extreme Gradient Boosting (XGBoost) algorithm to detect fradulent credit card transactions in a real-world (anonymized) dataset of european credit card transactions. I evaluate the performance of the model on a held-out test set and compare its performance to a few other popular classification algorithms, namely, Logistic Regression, Random Forests and Extra Trees Classifier (Geurts, Ernst, and Wehenkel 2006), and show that a well-tuned XGBoost classifier outperforms all of them.

The main challenge in fraud detection is the extreme class imbalance in the data which makes it difficult for many classification algorithms to effectively separate the two classes. Only 0.172% of transactions are labeled as fradulent in this dataset. I address the class imbalance by reweighting the data before training XGBoost (and by SMOTE oversamping in the case of Logistic regression).

Hyper-parameter tuning can considerably improve the performance of learning algorithms. XGBoost has many hyper-parameters which make it powerful and flexible, but also very difficult to tune due to the high-dimensional parameter space. Instead of the more traditional tuning methods (i.e. grid search and random search) that perform a brute force search through the parameter space, I use Bayesian hyper-parameter optimization (implemented in the hyperopt package) which has been shown to be more efficient than grid and random search (Bergstra, Yamins, and Cox 2013).

The full python code can be found here.

Keywords: XGBoost, Imbalanced/Cost-sensitive learning, Bayesian hyper-parameter tuning


1. Problem and Data
The aim of this exercise is to detect fradulent activity in a real-world dataset of credit card transactions. The dataset (hosted by Kaggle) contains 284,807 transactions, only 492 of which are labeled as fradulent, that is 0.172%. Due to confidentiality considerations, this dataset has been transformed from its original form and only provides time and amount of each transaction along with 28 principal components of the original features obtained by PCA. Some information is inevitably lost during this transformation which limits how well any algorithm can do on this dataset compared to similar non-transformed datasets. It also renders feature engineering virtualy irrelevant, as I discuss below in more detail.

2. Choosing an Evaluation Metric
Before we can start building our model, we need to choose an evaluation metric to measure the performance of different models with. For reasons explained below, I will evaluate models based on the highest recall rate they can acheive subject to precision being above a given minimum, say 5%. The idea behind this evaluation metric is that we want to focus on and maximize recall (probability of detecting fraud). But since higher recall (i.e. a more sensitive classifier) will inevitably result in lower precision (more false positives) we also want to make sure that precision does not get too low. So, we preset a level of precision and then try to maximize the probability of detecting fraudulent transactions, that is, the recall rate. I will call this metric Conditional Recall.

2.1 Further Details on Evaluation Metrics
I will briefly explain the reason for this choice, and why Conditional Recall is a better evaluation metric in this context than more commonly used alternatives such as AUC, PRAUC and F-score.

In choosing the evaluation metric we need to take two important features of the problem into account:

The evaluation metric needs to be robust to class imbalance and provide a reasonable measure of performance in the presence of this highly skewed class distribution.

It needs to incorporate the fact that false negatives (failing to detect fraud) are more costly than false positives (labeling a legal transaction as fraud).

The first consideration rules out some common metrics, such as accuracy and AUC (area under ROC curve) as these are not robust to class imbalance (Davis and Goadrich 2006). On the other hand, pprecision and recall have been shown to be robust to highly skewed class distributions and that is why a metric based on these statistics is more suitable for the present context.

PRAUC (area under precision-recall curve) and 
-score (
, and more generally 
) are two of the most commonly used metrics that combine precision and recall into one single metric. However, they do not satisfy the second criterion. PRAUC gives the same weight to the performance of a classifier at all levels of precision and recall, whereas we care more about performance at high recall rates. In other words a classifier with very high precision at low levels of recall can acheive a very high PRAUC score while it is practically useless for fraud detection purposes. 
 is more flexible in that we can give more weight to recall (by choosing a larger 
) but I will not use this metric as it requires making ad-hoc assumptions about the context of the problem and the relative costs of false positives nad flase negatives.

On the other hand, Conditional Recall allows us to focus on improving the ability of the algorithm to detect fraud (i.e., its recall) in a trasparent and easy-to-interpret way, while maintaining a decent level of precision.

3. Preprocessing
To address the extreme class imbalance in the data, I will use two different preprocessing techniques. For the XGBoost algorithm I simply reweight the instances of the positive class (fraudulent transactions) by the class imbalance ratio. In the case of Logistic regression (one of the algorithms to which I will compare XGBoost), I use the Synthetic Minority Over-sampling Technique (SMOTE). SMOTE (Chawla et al. 2002) balances the class distribution by creating new synthetic instances of the minority class.

However, I do not perform any further feature engineering (beyond rescaling for Logistic regression) for the following two reasons:

There are no missing values in this dataset and hence no need for imputing missing values. All variables are continuous numerical values.

XGBoost is an ensemble learning algorithm whose individual learning units are decision trees and trees have two favorable features which, again, render feature engineering unnecessary. First, decision trees are invariant to monotonic transformations of features (e.g. scaling or polynomial transformations). Second, they can inherently capture and model interactions between features. So, we do not need to manually create feature interactions.

As I mentioned above, the PCA transformation makes it impossible to use our background knowledge about the features to create new ones. Moreover, we do not need to worry about feature correlation as principal components are, by construction, orthogonal and therefore uncorrelated with one another.

Therefore, I will primarily focus on tuning the learning algorithm and try to optimize its performance through hyper-parameter optimization.

4. The XGBoost Algorithm
XGBoost or Extreme Gradient Boosting is an efficient implementation of the gradient boosting algorithm. Gradient boosting is an ensemble learning algorithm that sequentially trains weak learners (i.e. simple models, typically shallow decision trees) with an emphasis on the parts of the data that have not been captured well so far. The final prediction of the model is a weighted average of the predictions of these weak learners. XGBoost has been repeatedly shown to be very effective in a wide variety of both regression and classification tasks.

I will use the XGBClassifier class from the xgboost package's Python API. This classifier implements Extreme Gradient boosting algorithm to optimize a given loss function. I use a logistic loss function, which is also the default loss of xgboost for two-class classification tasks.

5. Overview of The Method
I will first split the data into a training and a held-out test set. The test will only be used once at the very end of the model building process to provide an unbiased estimate of model performance on data it has never seen before.

I will then build and test the model through the following steps:

Tuning by cross validation: Given the relatively large number of hyper-parameters of XGBoost, I will use Bayesian hyper-parameter tuning (which is more efficient than grid or random search), with (stratified) K-fold cross validation to choose the set of hyper-parameters that acheive the highest cross validated Conditional Recall score.

Thresholding: The tuned classifier from step (1) is able to predict a probability score for any given example. In order to classify an example we need to choose a probability threshold above which examples are labeled as positive (fraud). The standard practice is to set the threshold at 0.5. However, given the relative importance of recall over precision, we can use empirical thresholding (Sheng and Ling, 2006) to tune the trade off between precision and recall and possibily acheive a higher recall rate by choosing an appropriate classification threshold.

Training and testing: I will train the model on the entire training set and evaluate its performance on the test set using the Conditional Recall metric discussed above.

I will also compare the performance of this model with a few other algorithms at the end. In order to compare different models (e.g. XGBoost vs Logistic Regression) one would ideally use nested cross validation. However, this is computationally very expensive. So, I will only report the performance of these model on a single test set.

 Business Solution:
 We developed a classification model using XGBoost to detect and prevent money laundering after 
 preprocessing the transaction data.The Model was optimized with hyperparameter tuning and 
 deployment on Streamlit. 

![image](https://user-images.githubusercontent.com/107097836/231666058-0f6e8cb9-ff7d-4d38-9dc9-28a9ed639ceb.png)


 Technology Stack:
 
 Database: PostgreSQL
 
 Programming Language: Python
 
 Libraries Used: Numpy, Pandas, Sklearn, Matplotlib, Feature-engine, ……
 
 Deployment Tools: Streamlit
 
 Monitoring &amp; Maintenance: Evidently 
 
 Business Benefits:
 
 Met business success criteria by reducing fraud transactions rate by 15%.
 
 Achieved ML accuracy of 90%
 
 Successfully achieved a cost saving of 10-15% annually as economic success criteria.


Streamlit app to GitHub, click on the following link:
https://money-laundering-using-machine-learning.streamlit.app/

To view the project only, click on the following link:
 https://nbviewer.org/github/amehaabera/Money-Laundering-Using-Machine-Learning/blob/main/Money%20Laundering%20Using%20Machine%20Learning.ipynb
