**Anti-Money-laundering using Machine Learning**

  Financial institutions facing significant challenges in detecting and preventing money laundering due to 
 the large volume of transactions they handle, complex regulations to comply with, and the use of 
 deceptive tactics like shell companies in money laundering. Failure to prevent money laundering can 
 harm their reputation and ability to operate effectively.

![image](https://user-images.githubusercontent.com/107097836/231665827-17e8afaa-595b-4ece-b63f-8b17a95327a7.png)


Conclusion
This study demonstrated the potential of machine learning models in detecting money-laundering activities. XGBoost emerged as the most effective model, achieving an accuracy of 91.22% and a recall of 82%. Rigorous data preprocessing and handling of class imbalance were pivotal to the model’s success. The iterative approach, encompassing data preprocessing, parameter optimization, and model evaluation, played a pivotal role in achieving high accuracy. Key features such as oldbalanceOrg and newbalanceDest were identified as critical determinants of fraudulent behavior. Deploying the model as a Streamlit application provided a practical tool for real-time fraud detection, addressing the limitations of traditional rule-based systems. The following conclusions were drawn:
1.	Effectiveness of Machine Learning: Ensemble methods, particularly XGBoost, demonstrated high accuracy and recall, making them suitable for AML applications.
2.	Importance of Data Preprocessing: Rigorous preprocessing, including handling class imbalance with SMOTE, significantly improved model performance.
3.	Practical Deployment: The integration of the XGBoost model into a Streamlit app provided an interactive tool for real-time fraud detection, enhancing operational efficiency.
The proposed solution addresses key limitations of traditional rule-based systems, such as high false positive rates and lack of adaptability to evolving fraud patterns.


 Business Solution:
 We developed a classification model using XGBoost to detect and prevent money laundering after 
 preprocessing the transaction data.The Model was optimized with hyperparameter tuning and 
 deployment on Streamlit.. 

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
