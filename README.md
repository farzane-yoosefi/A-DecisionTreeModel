# In this page I explain the written code for begginers in the fiels of ML and demonstrate my knowledge to employers
NB : I have categorized the whole project into different functions.
At first we import the required libraries :

```python
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
```

**remember** : Any time that you build a model ,you have to already find out about the data.Therefore this function 
shows some information from the data like : length of data ,shape and first few rows.
**some explanations about the data** :
- There are 5 rows
- The first row contains the target value
- folowing 4 rows have a value between 1 to 5
- The first row : The object distance from the middle to the left
- The second row : The object wieght
- The third row : The object distance from the middle to the right
- The forth row : The object weight
  
 ```python
  def importdata():
    balance_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data')
    print("Dataset Length: ", len(balance_data))
    print("Dataset Shape: ", balance_data.shape)
    print("Dataset: ", balance_data.head())
 return balance_data
```

This function defines the Target and features of the data using slicing 

**In order to be able to slice the columns use ``.values``**

```python
def splitdataset(balance_data):

    # Separating the target variable
    X = balance_data.values[:, 1:5]
    Y = balance_data.values[:, 0]

    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=100)

    return X, Y, X_train, X_test, y_train, y_test
```

This function trains your model considering 'gini' as criteria
This is the first time we use Desicion Tree so let's figure out about their parameters 
- ``criterion`` : Here we define the criteria.It determines how the model picks the best question.
- ``max_leaf_nodes`` : Here we set a maximum for leaf nodes(or final answers) to prevent overfitting.
- ``max_depth`` : Here we define the maximum questions in one complete path from start to finish

In this function we train the model
- using ``'gini'`` criterion
- define a maximum of 5 final leaves
- a maximum of 3 pathes 

```python
 def train_using_Gini(X_train,Y_train):
     DecTree = DecisionTreeClassifier(criterion='gini',max_leaf_nodes=5,random_state=22,max_depth=3)
     trained_model = DecTree.fit(X_train,Y_train)
    return trained_model
```

**Now we do the same wiht the model but different with criterion: ``'entropy'``**
```python
def train_using_entropy(X_train,Y_train):
    Dec_tree = DecisionTreeClassifier(criterion='entropy',max_depth=5,max_leaf_nodes=5,random_state=3)
    trained_model = Dec_tree.fit(X_train,Y_train)
    return trained_model
```
**this function tests the trained model and returns predictions which I called it Y_hat accurding to its mathematical aspect**
I don't explain in datails becouse there does not envolve any important point about our main topic.But if you have any questions I ready to answer 🙂

```python
def prediction(X_test,trained_model ):
    Y_hat = trained_model.predict(X_test)
    print("Predicted values : ")
    print(Y_hat)
    return Y_hat
```
**After testing the data we have to assass the model accuracy**
```python
def Accuracy(Y_test,Y_hat ):
    accuracy = accuracy_score(Y_test,Y_hat)*100
    print(f'Accuracy : {accuracy_score}')
    confusion = confusion_matrix
    print(f'confusion_matrix:{confusion}')
    class_report =classification_report(Y_test,Y_hat)
    print(f"Classification : {class_report}")
```
**After all it is time to plot the DecisionTree**

for this purpose we need ``plot_tree`` and its parameters:
- ``classified_objects`` : It is our Decision Tree .In simple terms this is the output of trained model
- ``class_names`` : For this model class names are 'L','R','B'
- ``feature_names`` : These can be anything you choose relating to your data.In my case these are : 'Left_Distance', 'Left_weight' ,'Right_weight' , 'Right_Distance'
- ``filled`` : It takes True/False Which  colors the nodes based on majority class
- ``rounded`` : Rounds the final nodes for a better appearance.
  
```python
def plot_descision_tree(classified_objects,feature_names,class_names):
    plt.Figure(figsize=(15,15))
    plot_tree(classified_objects,class_names=class_names,feature_names=feature_names,filled=True,rounded=True)
    plt.show()
    return plot_tree
```
**Now that we have defined all functions that we need , it is time to use them.**

```python
if __name__ == '__main__':
    data = importdata()
    X,Y,X_train,X_test,y_train,y_test = splitdataset(data)

    #trin using GIni
    Gini = train_using_Gini(X_train,y_train)
 
    #train using entropt
    Entropy = train_using_entropy(X_train,y_train)

    #plot something
    plot_descision_tree(Gini,['Left_Distance', 'Left_weight' ,'Right_weight' , 'Right_Distance'],
                        ['L','B','R'])
    plot_descision_tree(Entropy,['Left_Distance', 'Left_weight' ,'Right_weight' , 'Right_Distance'],['L','R','B'])
```
**Now we see how acurate the model was :**
```python
Gini_result = prediction(X_test,Gini)
Entropy_result = prediction(X_test,Entropy)

Accuracy(y_test,Gini_result)
Accuracy(y_test,Entropy_result)
```








