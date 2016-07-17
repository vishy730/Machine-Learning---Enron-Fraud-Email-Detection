# Machine-Learning---Enron-Fraud-Email-Detection
Identify the Person of Interest from Enron Dataset using Machine Learning Techniques

#Udacity Nanodegree. Project 4. Identifying Fraud from Enron Email
##1. Dataset and goal of project
####Goal
The goal of this project is to identify, the person of interest(POI), who played fraud game using machine learning techniques.

####Dataset
We have 146 variables and 21 features in this dataset to test. Here is one of the variables in the dataset.

	[{"SKILLING JEFFREY K":{
				'salary': 1111258, 
				'to_messages': 3627, 
				'deferral_payments': 'NaN', 
				'total_payments': 8682716, 
				'exercised_stock_options': 19250000, 
				'bonus': 5600000, 
				'restricted_stock': 6843672, 
				'shared_receipt_with_poi': 2042, 
				'restricted_stock_deferred': 'NaN', 
				'total_stock_value': 26093672, 
				'expenses': 29336, 
				'loan_advances': 'NaN', 
				'from_messages': 108, 
				'other': 22122, 
				'from_this_person_to_poi': 30, 
				'poi': True, 
				'director_fees': 'NaN', 
				'deferred_income': 'NaN', 
				'long_term_incentive': 1920000, 
				'email_address': 'jeff.skilling@enron.com', 
				'from_poi_to_this_person': 88
				}
	}]

####Outliers
After observing the data, we could see that Total and The Travel Agency in the Park were considered as outliers and removed them.

##2. Feature selection process
####I selected the following features after few iterations.
`exercised_stock_options`  `shared_receipt_with_poi`  `fraction_from_poi`  `expenses`  `other` `salary`

####New features
In addition I create three features which were considered in course:
* `fraction_from_poi` fraction of messages to that person from a POI
* `fraction_to_poi` fraction of messages from that person to a POI

Another feature, Messages to current person from specific email addresses, which belong to outliers such as highest salary. 
* `from_specific_email` 


For feature selection process, have iterated several iterations. 
Based on intuition and from course lesson's exercises, have selected some features and removed the outliers. I have selected Decision Tree Classifier as the algorithm for finding the POIs.
Later used feature importance method to optimize features for this dataset. 

As a result I’ve received the following feature importances:

	Rank of features
	0.225391 : other
  0.217031 : exercised_stock_options
  0.195511 : shared_receipt_with_poi
  0.185413 : expenses
  0.145300 : fraction_from_poi
  0.031354 : salary

The new features created fraction_from_poi seems not that significant and other features fraction_to_poi and from_specific_email were zero and have excluded from the algorithm.

##3. Pick an algorithm
I tried the Naive Bayes, SVM and Decision Trees algorithms. 

####All results of examination I included in the following table

|Algorithm|Accuracy|Precisions|Recall|
|:---|---|---|---|
|**Naive Bayes**|0.83871|0.39049|0.23000|
|**Decision Trees**|0.85986|0.51180|0.41200|
|**SVM**|-|-|-|

SVM algorithm returned the next error :
```
Got a divide by zero when trying out: SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
```

####Final algorithm
Based on the above numbers, I zeroed down Decision Tree Classifier as my final algorithm.

##4. Tune the algorithm
The purpose of tuning is to get the best results by changing various parameters of the algorithm.

####GridSearchCV
I apply GridSearchCV to tune the following parameters

|Parameter          |Settings for investigation |
|:------------------|:--------------------------|
|min_samples_split	| [2,6,8,10]                | 
|Splitter	        | (random,best)             |
|max_depth	        | [None,2,4,6,8,10,15,20]   |

As a result, I received better performance with `min_samples_split` = '12' and `Splitter` = 'best' `and max_depth` = '6'

##5. Validation
According to [sklearn documentation][sklearn_mistake] one of the main and classical mistakes in validation is using the same data for both training and testing. 
>Learning the parameters of a prediction function and testing it on the same data is a methodological mistake: 
>a model that would just repeat the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data. This situation is called overfitting.

To validate my analysis I used [stratified shuffle split cross validation][StratifiedShuffleSplit] developed by Udacity and defined in tester.py file

##6. Evaluation metrics
I used precision and recall evaluation metrics to estimate model.
Final results can be found in table below

|Metric|Value|
|:----|:----|
|**Precision**|**0.52166**|
|**Recall**|**0.41550**|
|Accuracy |0.86207|
|True positives |831|
|False positives|762|
|False negatives|1169|
|True negatives|11238|
	
####Conclusion
Precision and Recall have almost identical values and both higher than .3. Thus, project goal was reached.
Precision 0.51180 means when model detect person as POI it was true only in 51% cases. 
At the same time Recall = 0.41200 says only 41% of all POIs was detected.

We have very imbalanced classes in E+F dataset. In addition, almost half of all POIs weren't included in dataset. 
In such conditions result we received good enough, but it's not perfect, of course.

#Algorithm outputs log
####STEP1. Init stage. Select classificator
#####features_list: 
	features_list = ['poi',
                 'fraction_to_poi',
                 'fraction_from_poi',
                 'from_specific_email',
                 'from_messages',
                 'exercised_stock_options',
                 'shared_receipt_with_poi',
                 'expenses',
                 'other',
                 'bonus',
                 'salary',
                 'total_stock_value'] 
				 
#####Classificator:
	clf = tree.DecisionTreeClassifier()
#####Metrics:
Accuracy: 0.83600	Precision: 0.41052	Recall: 0.33950	F1: 0.37165	F2: 0.35167
	Total predictions: 14000	
	True positives:  679	False positives:  975	
	False negatives: 1321	True negatives: 11025
#####Classificator:
	clf = GaussianNB()
#####Metrics:
	Accuracy: 0.83871	Precision: 0.39049	Recall: 0.23000	F1: 0.28949	F2: 0.25060
	Total predictions: 14000	
	True positives:  460	False positives:  718	
	False negatives: 1540	True negatives: 11282

####STEP2. Select features by Decision Trees feature_importances_
#####feature_importances_
	Rank of features
	0.284728 : other
0.217361 : exercised_stock_options
0.190413 : expenses
0.146331 : fraction_from_poi
0.129887 : shared_receipt_with_poi
0.031281 : bonus
0.000000 : fraction_to_poi
0.000000 : from_specific_email
0.000000 : from_messages
0.000000 : salary
0.000000 : total_stock_value

#####Metrics after optimizing
	Accuracy: 0.84029	Precision: 0.43873	Recall: 0.42250	F1: 0.43046	F2: 0.42565
	Total predictions: 14000	
	True positives:  845	False positives: 1081	
	False negatives: 1155	True negatives: 10919

####STEP3. Tune the algorithm
#####features_list
	features_list = ['poi',
                 'salary',
                 'fraction_from_poi',
                 'exercised_stock_options',
                 'shared_receipt_with_poi',
                 'expenses',
                 'other'] 
#####best estimator:
	DecisionTreeClassifier(compute_importances=None, criterion='gini',
            max_depth=6, max_features=None, max_leaf_nodes=None,
            min_density=None, min_samples_leaf=1, min_samples_split=12,
            random_state=None, splitter='best')
#####Metrics after tuning (BEST CHOISE!)
	Accuracy: 0.85986	Precision: 0.51180	Recall: 0.41200	F1: 0.45651	F2: 0.42872
	Total predictions: 14000	
	True positives:  824	False positives:  786	
	False negatives: 1176	True negatives: 11214

####STEP4. Change features by hand (examine only email features)
	features_list = ['poi',
                 'fraction_from_poi',
                 'fraction_to_poi',                 
                 'exercised_stock_options',
                 'shared_receipt_with_poi'] 
#####Metrics 
	Accuracy: 0.81750	Precision: 0.42624	Recall: 0.27450	F1: 0.33394	F2: 0.29554
	Total predictions: 12000	
	True positives:  549	False positives:  739	
	False negatives: 1451	True negatives: 9261
	
####STEP5. Tune parameters by hand
#####parameters
	clf = tree.DecisionTreeClassifier(random_state=42, min_samples_split=2,max_depth=2, splitter='best')
#####Metrics 
	Accuracy: 0.83550	Precision: 0.37947	Recall: 0.23850	F1: 0.29291	F2: 0.25764
	Total predictions: 14000	
	True positives:  477	False positives:  780	
	False negatives: 1523	True negatives: 11220

So I finally choosen the parameters and features that I have used in Step 4 as they have given the best results.
	
References
- [Documentation of scikit-learn 0.15][1]
- [sklearn tutorial][2]
- [Recursive Feature Elimination][3]
- [Selecting good features – Part I: univariate selection][4]
- [Cross-validation: the right and the wrong way][6]
- [Accuracy, Precision and Recall(in Russian)][6] 

[1]: http://scikit-learn.org/stable/documentation.html
[2]: http://amueller.github.io/sklearn_tutorial/
[3]: http://topepo.github.io/caret/rfe.html
[4]: http://blog.datadive.net/selecting-good-features-part-i-univariate-selection/
[5]: https://www.kaggle.com/c/the-analytics-edge-mit-15-071x/forums/t/7837/cross-validation-the-right-and-the-wrong-way
[6]: http://bazhenov.me/blog/2012/07/21/classification-performance-evaluation.html
[StratifiedShuffleSplit]: http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
[sklearn_mistake]: http://scikit-learn.org/stable/modules/cross_validation.html 
