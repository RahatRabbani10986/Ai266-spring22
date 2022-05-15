# Ai-106394 : Project #
### PROJECT MEMBERS ###
StdID | Name
------------ | -------------
**10986** | **Rahat Rabbani** 
11300 | salah shakeel
10973 | Aiman Akmal
11081 | Ahmed Khan
11193 | Huzaifa Shakeel

## Problem ##
1. Laplace smoothing was not an actual model we had to use multinationalnb with alpha 0.1
2.lidstone smoothing was achieved by using multinomialnv with alpha 1
3.we had to set up appropriate neighbors value  we figured it out with hit and try
4.k fold validation code was complex we solved this problem by using it through a library
## Assignemnt Description ##
train 5 models and check scores on kaggle

## Screenshot of highest score on kaggle ##
![Screenshot_1](https://user-images.githubusercontent.com/)
![Screenshot_2](https://user-images.githubusercontent.com/)
![Screenshot_3](https://user-images.githubusercontent.com/)
![Screenshot_4](https://user-images.githubusercontent.com/)
![Screenshot_5](https://user-images.githubusercontent.com/)


## Models Used ##
    a. MultinomialNB (laplace and lidstone)
    b. perceptron
    c. SVM
    d. KNN

### KNN ###
The k-nearest neighbors (KNN) algorithm can be used to solve both classification and regression problems. in knn we have used 3 as nearest neighbor to better satisfy our need.(performed by leader)

### SVM ###
Support Vector Machine (SVM) algorithm can be used for both classification or regression challenges. (performed by member1)

### Linear Regression ###
Linear Regression algorithm is used to performs a regression task. (performed by member2)

### MultinomialNB ###
MultinomialNB algorithm is used in Text Classification, Spam filtering and Sentiment Analysis. (laplace smoothing by member3 and lidstone smoothing by member4)

## References ##
- Documentation of python : https://docs.python.org/3/
- Documentation of scikit-learn : https://scikit-learn.org/stable/user_guide.html
- Explanation of MulitinomilNB with laplace and lidstone smoothing : https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html
- Explanation of SVM : https://scikit-learn.org/stable/modules/svm.html
- Explanation of KNN : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
- Explanation of Perceptron : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
