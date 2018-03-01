
# coding: utf-8

# Author: Aditya Joshi.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data_train = pd.read_csv('/Users/adityajoshi/DataSets/Datasets/Titanic/Titanic_train.csv')
data_test = pd.read_csv('/Users/adityajoshi/DataSets/Datasets/Titanic/Titanic_test.csv')
data_train.sample(3)


# In[3]:


#Embarked vs Survived
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train);
plt.show()


# In[4]:


#Pclass vs Survived with gender classification
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data_train,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"])


# In[5]:


plt.show()


# In[6]:


#No missing val in Survived
data_train['Survived'].isnull().sum()


# In[7]:


#Drop Survived from train
survived = data_train['Survived']
data_train.drop('Survived',axis = 1,inplace = True)


# In[8]:


#Passenger ID not important while training
data_test['PassengerId'].isnull().sum()


# In[9]:


Pssid = data_test['PassengerId']


# In[10]:


train_test = pd.concat([data_train,data_test],keys = ['train','test'],names = ['dataset','index'])
train_test.head()


# In[11]:


train_test.info()


# In[12]:


#Drop 'PassengerId','Ticket','Name'
train_test.drop(['PassengerId','Ticket','Name'],axis = 1,inplace=True)
train_test.info()


# In[13]:


train_test.isnull().sum()


# In[14]:


1014 / float(1307)


# In[15]:


#More than 75% missing vals in Cabin
train_test.drop(['Cabin'],axis=1,inplace=True)


# In[16]:


#Fill missing vals in other cols
avgAge = train_test['Age'].mean()


# In[17]:


train_test['Age'] = train_test['Age'].fillna(avgAge)


# In[18]:


avgAge


# In[19]:


train_test['Embarked'].describe()


# In[20]:


train_test['Embarked'] = train_test['Embarked'].fillna('S')


# In[21]:


train_test['Fare'] = train_test['Fare'].fillna(train_test['Fare'].mean())


# In[22]:


#No missing val
train_test.isnull().sum()


# In[23]:


train = train_test.loc['train']
test = train_test.loc['test']


# In[24]:


train['Survived'] = survived


# In[25]:


train.head()


# In[26]:


train.shape


# In[27]:


survived.shape


# In[28]:


#EDA 

plt.figure(figsize=(12,6))
g = sns.countplot(x = survived,data=train,hue='Sex')
for p in g.patches:
    g.annotate("%d" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', xytext=(0, 8), textcoords='offset points')


# In[29]:


plt.show()


# In[30]:


1-train.groupby(by = 'Sex')['Survived'].mean()


# In[31]:


#Ratio of Males lost relative to the females lost
.81/.25


# In[32]:


plt.figure(figsize=(12,6))
g = sns.countplot(x = 'Pclass',data = train)
for p in g.patches:
        g.annotate("%d" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', xytext=(0, 8), textcoords='offset points')


# In[33]:


plt.show()


# In[34]:


1-train.groupby(by = 'Pclass')['Survived'].mean()


# In[35]:


1-train.groupby(by = 'Embarked')['Survived'].mean()


# In[36]:


plt.figure(figsize=(12,6))
g = sns.countplot(x = 'Embarked',data = train,hue = 'Pclass')
for p in g.patches:
        g.annotate("%d" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', xytext=(0, 8), textcoords='offset points')
plt.show()


# In[37]:


g = sns.FacetGrid(train,row='Pclass',col='Embarked',size=5,aspect=0.7)
g = (g.map(sns.countplot,'Survived',hue='Sex',data=train,palette='Set1')).add_legend()
for ax in g.axes.ravel():
    for p in ax.patches:
        ax.annotate("%0.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', xytext=(0, 8), textcoords='offset points')


# In[38]:


plt.show()


# In[39]:


ECS = pd.DataFrame(train.groupby(by = ['Embarked','Pclass'])['Sex'].value_counts())


# In[40]:


ECS_per = pd.DataFrame(1 - train.groupby(by = ['Embarked','Pclass','Sex'])['Survived'].mean())


# In[41]:


#ECS_per = pd.DataFrame(round(1 - train.groupby(by = ['Embarked','Pclass','Sex'])['Survived'].mean(),3))
#ECS_per = pd.DataFrame(1 - train.groupby(by = ['Embarked','Pclass','Sex'])['Survived'].mean(),3)
ECS_lost = pd.DataFrame(train.groupby(by = ['Embarked','Pclass'])['Sex'].value_counts() - train.groupby(by = ['Embarked', 'Pclass', 'Sex'])['Survived'].sum())
ECS = pd.concat([ECS, ECS_lost,ECS_per], axis = 1)
ECS.columns = ['Total number of passengers', 'Number of passengers lost','Percentage of passengers lost']
ECS = ECS.style.set_properties(**{'text-align': 'right'})
ECS


# In[42]:


#Age
train['Age'].describe()


# In[43]:


plt.figure(figsize=(16,6))
sns.distplot(train['Age'],kde=False,bins=150)


# In[44]:


plt.show()


# In[45]:


train.groupby(by = 'Sex')['Age'].describe()


# In[46]:


plt.figure(figsize=(12,6))
sns.boxplot(x = 'Sex',y = 'Age',data = train)
plt.show()


# In[47]:


#Fare


# In[48]:


train['Fare'].describe()


# In[49]:


plt.figure(figsize=(16,6))
sns.distplot(train['Fare'],hist_kws={'edgecolor':'k'}, kde=False,bins=150)
plt.show()


# In[50]:


#Applying ML algorithms.

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import statsmodels.api as sm
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV


# In[51]:


train_get_dummy = pd.get_dummies(columns=['Sex', 'Pclass', 'Embarked'], drop_first = True, data = train)


# In[52]:


train_get_dummy.head()


# In[53]:


def forward_stepwise_selection(df):
    p = len(df.columns)
    X = df.drop('Survived',axis = 1)
    columns = list(X.columns)
    y = df['Survived']
    best_cols_global = []
    for col1 in columns:
        max_score = -1
        for col2 in columns:
            model = LogisticRegression()
            if col2 not in best_cols_global:
                cols = best_cols_global[:]
                cols.append(col2)
                model.fit(X[cols],y)
                score = model.score(X[cols],y)
                if score > max_score:
                    max_score = score
                    best_col = col2
        if best_col not in best_cols_global:
            best_cols_global.append(best_col)
        print(best_cols_global, max_score)
        model = LogisticRegression()
        mean_score = cross_val_score(model,X[best_cols_global],y,cv = 5).mean()
        print('CV mean score is ', mean_score)


# In[54]:


forward_stepwise_selection(train_get_dummy)


# In[55]:


def summary_results(df):
    x_train, x_test, y_train, y_test = train_test_split(df.drop('Survived',axis = 1),df['Survived'],test_size = 0.3,random_state = 100)
    k_max = -10
    mean_score_max = -10
    for kk in range(1,20,2):
        KNN_model = KNeighborsClassifier(n_neighbors=kk)
        mean_score = cross_val_score(KNN_model,df.drop('Survived',axis = 1),df['Survived'],cv = 5).mean()
        if mean_score > mean_score_max:
            mean_score_max = mean_score
            k_max = kk
    print('The best K for KNN using 5 fold CV is',k_max)
    DC = DummyClassifier(strategy='most_frequent')
    LR = LogisticRegression()
    LDA = LinearDiscriminantAnalysis()
    QDA = QuadraticDiscriminantAnalysis()
    KNN = KNeighborsClassifier(n_neighbors=k_max)
    VC = VotingClassifier(estimators=[('lr',LR),('KNN',KNN)],voting='soft')
    classfiers = [DC,LR,KNN,VC]
    list_of_classifiers = ['Null Classifier','Logistic Regression','K-Nearest Neighbors','Voting Classifier']
    max_len = max(list_of_classifiers,key=len)
    
    
    i = 0
    for model in classfiers:
        model.fit(x_train, y_train)
        pred = model.predict(x_test) 
        Con_mat = confusion_matrix(y_test, pred)
        acc = (Con_mat[0,0]+Con_mat[1,1])/float((Con_mat[0,0]+Con_mat[0,1]+Con_mat[1,0]+Con_mat[1,1]))
        mean_score = cross_val_score(model, df.drop('Survived', axis = 1), df['Survived'], cv = 5).mean()
        diff = len(max_len)-len(list_of_classifiers[i])+5
        print('{}{}{:.3f}\t{:.3f}'.format(list_of_classifiers[i], ' '*diff, acc, mean_score))
        #print(list_of_classifiers[i], acc)
        i += 1


# In[56]:


summary_results(train_get_dummy[['Survived','Sex_male','SibSp','Pclass_3','Age','Embarked_S']])


# In[57]:


test_get_dummies = pd.get_dummies(test,columns=['Sex','Pclass','Embarked'],drop_first=True)


# In[58]:


test_get_dummies = test_get_dummies[['Sex_male','SibSp','Pclass_3','Age','Embarked_S']]


# In[59]:


LR = LogisticRegression()
KNN = KNeighborsClassifier(n_neighbors=11)
best_classifier = VotingClassifier(estimators=[('lr',LR),('KNN',KNN)],voting='soft')


# In[60]:


best_classifier.fit(train_get_dummy[['Sex_male','SibSp','Pclass_3','Age','Embarked_S']],train_get_dummy['Survived'])


# In[61]:


PssID_df = pd.DataFrame(Pssid,columns=['PassengerId'])
PssID_df['Survived'] = best_classifier.predict(test_get_dummies)
PssID_df.to_csv('/Users/adityajoshi/DataSets/Datasets/Titanic/result.csv',index = False)


# Accuracy on Kaggle: 0.75119
