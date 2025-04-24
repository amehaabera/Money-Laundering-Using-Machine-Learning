**Anti-Money-laundering using Machine Learning**

The proliferation of financial crimes, particularly money laundering, demands advanced detection systems capable of identifying suspicious transactions amidst vast, imbalanced datasets. This thesis investigates the efficacy of machine learning algorithms in detecting money laundering activities, focusing on optimizing model performance through iterative experimentation, data preprocessing, and hyperparameter tuning. Five algorithms—Logistic Regression, K-Nearest Neighbors, Random Forest, XGBoost, and Support Vector Machines—were evaluated, with XGBoost emerging as the superior model due to its robustness in handling class imbalance and complex transactional patterns.
A review of the data showed that the data was unbalanced as there were only 913 suspicious transactions in 572,196 flagged suspicious transactions. The study highlights the critical role of preprocessing techniques, particularly SMOTE (Synthetic Minority Oversampling), in addressing class imbalance. By balancing datasets, SMOTE improved XGBoost’s recall to 99.99%, reducing false negatives to just 14 cases. Iterative experiments further revealed that transaction balances (Old Balance Org, New Balance Orig) and transaction type (TRANSFER, CASH_OUT) were the most influential features, aligning with patterns identified through exploratory data analysis. While Random Forest achieved near-perfect precision (99.29%), its high false-negative rate rendered it less practical for real-world deployment.
The findings underscore the limitations of accuracy as a performance metric in imbalanced contexts, advocating for the F1 score as a more reliable indicator. The thesis concludes that XGBoost, combined with SMOTE, offers a scalable solution for financial institutions to mitigate fraud risks. A Streamlit-based deployment demonstrated real-time detection capabilities. This research contributes a framework for deploying machine learning in anti-money laundering (AML) workflows, balancing detection efficacy with operational feasibility.


![image](https://user-images.githubusercontent.com/107097836/231665827-17e8afaa-595b-4ece-b63f-8b17a95327a7.png)


Conclusion
This study demonstrates that XGBoost, combined with synthetic minority oversampling (SMOTE), is the most effective approach for detecting money laundering transactions. By addressing class imbalance and prioritizing recall, XGBoost achieved near-perfect detection rates (e.g., 99.96% recall in Iteration 3) while maintaining high precision (96.82–99.58%). The algorithm’s adaptability to preprocessing techniques, such as subsampling and hyperparameter tuning, underscores its superiority over alternatives like Random Forest, which prioritized precision at the cost of missing critical fraud cases (e.g., 566 undetected frauds in Iteration 3).
Key features such as transaction types (TRANSFER/CASH_OUT), balance dynamics (New/Old Balance Orig/Dest), and temporal patterns (Transaction Day/Hour) emerged as the most discriminative indicators of suspicious activity, aligning with findings from exploratory data analysis (EDA). While high accuracy scores (e.g., 99.82% in Iteration 3) were achieved on balanced data, the study highlights the limitations of accuracy in imbalanced real-world scenarios, advocating for F1-score and confusion matrices as more reliable metrics.
The deployment of the model as a Streamlit application provided a practical tool for real-time fraud detection, addressing the limitations of traditional rule-based systems (e.g., high false positives and rigidity to evolving fraud patterns).
 

![image](https://user-images.githubusercontent.com/107097836/231666058-0f6e8cb9-ff7d-4d38-9dc9-28a9ed639ceb.png)


 Technology Stack:
 
 Database: PostgreSQL
 
 Programming Language: Python
 
 Libraries Used: Numpy, Pandas, Sklearn, Matplotlib, Feature-engine, ……
 
 Deployment Tools: Streamlit
 
 Monitoring &amp; Maintenance: Evidently 
 
Streamlit app to GitHub, click on the following link:
https://money-laundering-using-machine-learning.streamlit.app/

To view the project only, click on the following link:
 [https://nbviewer.org/github/amehaabera/Money-Laundering-Using-Machine-Learning/blob/main/Money%20Laundering%20Using%20Machine%20Learning.ipynb
](https://nbviewer.org/github/amehaabera/Money-Laundering-Using-Machine-Learning/blob/main/Final%20Money%20Laundering%20Using%20Machine%20Learning.ipynb)
