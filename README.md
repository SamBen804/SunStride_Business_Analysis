![Sunrise](images/Sunrise.jpeg)

# SunStride_Business_Analysis

## Overview

SunStride is an adventure travel company for teens that sends students all over the world each summer season. SunStride needs a system for projecting how many students they can expect to return next summer so they can approriately allocate funds and pay their outfitters' non-refundable deposits almost a year in advance. I built the company a predictive model using a random forest classifier and it resulted in a model that is able to predict a student will return with a precision score of ~.85. Based on this result, SunStride is able to make a confident determination of how many students will return and how much money they may need to spend on marketing to fill in the rest of the spaces with new students. The model identified the feature "Years at camp" as the most impactful when the model was making splits. This indicates that if SunStride is able to retain a student for at least a second summer season then they are significantly more likely to continue seeing that student return for future summers. To retain students I recommend SunStride focus any marketing spend on finding new, young customers who have many years left to return and keeping these students as engaged as possible after they have signed up. 

## Business Problem

Because the company only operates its trips in the summer, its cash flow is cyclical annually with a major influx in August when enrollment opens and then it tapers off throughout the rest of the year. This inflow of cash is crucial, as it allows the company to hire and pay the outfitters that will run the next seasons' trips (many of whom require a majority of the payment up front and a year in advance to hold the spots!). SunStride needs a more robust system for projecting how many students they can expect to return the following summer so they can project how many spaces to hold (and pay for) for activities the next season. 

The final model I built SunStride uses random forest classification to sort students into two categories: returning (designated with a 1) or not (a 0). The model takes into account features like the student's grade, gender, and their overall rating of the trip they just finished (rated on a scale from 1-6). Initially, the models I built were drastically overfitting to the training data - as many tree-like classifiers tend to do if left un-tuned. With proper tuning through grid-searching and additional model iterations, the final model achieved an accuracy score of .86, a precision score of .85 and a recall score of .84. 

For this analysis, I am focused on the precision score which represents the ratio of false positives predicted by the model. A false positive means the model is predicting that a student will return, when they do not actually re-enroll the next season. The lower the precision score (between 0 and 1.0) the higher the number of false positives predicted. If SunStride were to use a model that predicted many false positives then they may allocate funds incorrectly, as they are projecting a higher student return than the company will actually see. Of course there are also new students that come on board each season, but it requires less marketing spend and less effort in the onboarding process to retain an existing student than it does to onboard a new student/family. 

A false negative, on the other hand, which is tracked best through the recall score, represents our models ability to predict a student will not return when they actually do. This is not necessarily ideal either for SunStride, as it represents money left on the table or it may miss earning opportunities by not having space for these students, but ultimately it does not cause any immediate negative impact on the business if not enough spots are offered because no money is exchanging hands with outfitters in this case.

## Data Understanding

The data was provided by SunStride from their customer management system. This system houses all the demographic data fro each custemr along with the post-trip ratings and opinions that each student reported after travelling with the company. Specifically, we are analyzing is the 2022 summer's student data, because we are able to identify and track whether or not these students returned for the 2023 season. The target variable is the student's 2023 season's status, which was distilled down to represent whether the student returned (Enrolled) or did not (which could have been represented by a few other designations that all meant the student did not travel with SunStride again). 

## Data Preparation

The features included in the dataset are a mix of both categorical and numeric varibles. Significant pre-processing was needed to clean and prepare the data for final modeling. During this cleaning stage, many columns were dropped from the dataset for various reasons (such as too many unique values to impactful in a predictive model, etc), and all of these reasons are documented in the "Final_Data_Cleaning" notebook linked here along with detailed descriptions of all the features that are included in the final model.

Some additional pre-processing steps included the imputation of missing column values and the OneHotEncoding of the categorical freatures. The numeric features in this case were not scaled, because the final model does not take the measure of distance into account, therefore the varying scales of the numeric features does not negatively impact the model's performance.

### Train-test Splitting
> - Before any modeling can begin, I will split the training and testing data. Performing this step now prevents data leakage by keeping the testing data hidden throughout the entire modeling process until a final model is chosen. 

### Baseline Understanding: Analyzing 'target' Column and Building a DummyClassifier
> - First I will check the value counts in the taget column and normalize the output to see the ratio of students that returned to the overall dataset. 
> - Then I will build a dummy model, which should result in scores that mirror the value I discovered in the normailzed value_counts of the 'target' column, becasue the dummy model will only pick the positive (1) value class. Of course this inital model will only act to represent a starting point for analysis, as it would not be useful to SunStride to have so many false positive predictions. 

> - I expect to see the ~.48 value for the positive class (designating the model is predicting that a student will return) to be repeated in the scoring for the dummy model:

> - This model's precision score represents the baseline that I will seek to improve through various modeling techniques and tuning. My goal moving forward will be to improve upon this ~.48 precision score. I plan to continue to include the other scores in my analysis, however my focus on optimizing the precision score will remain. See futher explanation below:

#### Focus on Precision Score and Reducing False Positives:
> The precision score represents the 'False Positives' generated by the model, this means the model is predicting that a student will re-enroll (designated with a 1 in the confusion matrix above) when they in fact do not re-enroll the following summer (designated by a 0 marker). 
> - The precision score represents all of the students that were incorrectly predicted to re-enroll the next summer when they do not in fact choose to travel with SunStride again. This mistaken prediction could cause the SunStride HQ to allocate funds for this higher expected enrollment, over-budgeting and ultimately being left with unfilled trips.
> - For many of the outfitters that SunStride works with, a flat rate is charged regardless of the trip capacity, thus causing SunStride to have to take a loss for these unfilled spots.
> - The recall score, in this case, represents the model's generation of False Negatives. 
>> - A False Negative in this case represents the instances where the model has designated a student will NOT re-enroll the next summer but they actually do.
>> - Yes, this figure could represent money left on the table, defined as an opportunity for SunStride to fill a trip spot that the company has not allocated resources for; however, SunStride does not ultimately lose any money out of pocket in these instances and would be able to add the student to the Waitlist for the next summer, which can build anticipation for students and families for the next season.

### OneHotEncoder
> - In order to use the categorical feature columns in the classifier models I will build, I need to first OneHotEncode out each of the columns' categorical variables. This means that I will be creating a matrix of all binary values that the classifier will be able to read and use to make splitting decisions.
> - For example: the OneHotEncoder dealing with the 'Gender' column which contains either 'Male' or 'Female' will split this column into two new columns for each record in the dataset. If the student's gender is designated male before the split then the resulting 'Gender_Male' column generated by the encoder will contain a value of 1 and the 'Gender_Female' column for this same record will contain a 0 and so on. 

## Modeling:

### DecisionTree Classifier: Model 1
> - The first iteration of the DecisionTree will use all default settings from sklearn and then later I will begin to tune these settings. For reference, the sklearn documentation for a DecisionTreeClassifier is provided here: [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.feature_importances_)


DecisionTree Iteration 1 Initial Interpretations:
> - The model appears to be very overfit, as evidenced by all scores of 1.0
> - When I next cross-validate the training data, I will likely see a lower value for all the scores, as this model isn't likely to remain perfect.

#### Cross-Validation on DecisionTree Iteration 1:
> - The 'cv' hyperparameter is set to 5, which is the default and this designates how many folds or splits within the training data will occur during the cross validation process.
> - The next printout cell highlights how each of these folds represents a different sample of data from the training set. As you can see in the output, each of the printed functions returned 5 cross-validated scores with different values due to the separate data that was assigned to each fold.
> - For ease of interpretation, the cell after prints the rounded mean of the all 5 of each of the scores.

#### DecisionTree Model 1 Interpretations:
> - After cross-validating, the first iteration of the DecisionTree model was definitely overfit, as evidenced by the lower scores for Accuracy, Precision, and Recall.
> - Overall a precision score of .81 is not bad, but it certainly leaves some room for improvement. I will further analyze the features contributing to this inital high score so I can later check if these features remain as significant throughout later model iteraions.

#### Analyzing Model 1 Feature Importances:
> - Identifying which features contributed the most to the DecisionTree's classification process.

It appears that Model 1 used the 'Years at camp' feature the most when making splitting decisions. This feature appears to have had a signigicantly higher impact on splitting decisions than any of the next 9 features in this list. This result is not necessarily shocking, as this feature is a numeric value that increases as the student travels with the company multiple summers but it does highlight the fact that it appears if the company is able to get a student back for a second summer then the likelihood that the student will return for 3 or more summers significantly increases.
The features for 12th and 11th grade, also make sense to be seen in high in this ranking for the opposite reason from the 'Years at camp' feature, as students in the 12th grade have travelled for their last summer due to the age cut-off and therefore if the model is reading a record of a student in 12th grade then they are guaranteed to not return the following summer. 11th grade students appear to be less likely to return the following year too as they would be returning for the summer after their 12th grade or Senior year of highschool.
The remaining features are interesting as well and I will keep them in mind as I build more models and will compare my final results to see if the top ten features listed remain the same as this initial feature interpretation.

### Grid Searching for Model Type 1: DecisionTree Classifier
> - I preformed a grid search next to identify the optimal hyperparameters for the DecisionTree model that would result in the best precision score.
> - I included all the default hyperparameter settings so the grid search will compare these options used in Model 1 as it iterates through making each of the new models with different hyperparameters tuned. 
> The two hyperparameters I'm focusing my grid search on:
>> - **'max_depth'**: this defines a hard stopping point for splits the decision tree is allowed to make. The default is None because, left unchecked a decision tree will just keep splitting until it creates pure leaf nodes. By setting a max_depth the tree is better able to generalize to new data because it does not have the free range to overfit to training data because we are setting a hard stop parameter.
>> - **'min_samples_leaf'**: this hyperparameter defines how many samples a leaf node can have at a minimum. The defualt is 1 because that is the purest form of a leaf node. By increasing the minimum sample per leaf, the tree is forced into another hard stop when it reaches the minimum specified. Similar to the max_depth parameter, this hyperparameter is used to reduce overfitting and it generates a model that generalizes better to unseen data, which is waht we are hoping for!


> - The model marginally improved the precision score with these tuned settings and I will look into it further.
> - According to the grid search, the hyperparameter settings that optimize the precision score are a max_depth of 25 in conjunction with a minimium of 3 samples per leaf node. These settings will be input into the next model iteration.

### DecisionTree Classifer Tuned: Model 2
> - Based on the grid search results, I have tuned the DecisionTree to have the hyperparameters that optimize the prescision score:
