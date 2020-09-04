#Flask Libraries
import matplotlib
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

#Data Analysis Libraries
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score
import sys
warnings.filterwarnings('ignore')

#Data Analyzing
def KNN(Ks,independents,dependent,sp):
    print(Ks,independents,dependent,sp)
    from sklearn.neighbors import KNeighborsClassifier
    X_train,X_test,y_train,y_test=train_test_split(df[independents],df[dependent],test_size=(1-(sp/100)),random_state=4)
    mean_acc=np.zeros((Ks))
    std_acc=np.zeros((Ks))
    for n in range(1,Ks+1):
        neigh=KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
        yhat=neigh.predict(X_test)
        mean_acc[n-1]=metrics.accuracy_score(y_test,yhat)
        std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
    for i in range(Ks):
        if mean_acc[i]==max(mean_acc):
            break
    K=i+1
    neigh=KNeighborsClassifier(n_neighbors=K).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    FS=f1_score(y_test,yhat,average='weighted')
    ACC=100*FS
    return K,ACC

def SVM(tech,independents,dependent,sp):
    import scipy.optimize as opt
    from sklearn import svm
    X_train,X_test,y_train,y_test=train_test_split(df[independents],df[dependent],test_size=(1-(sp/100)),random_state=4)
    clf=svm.SVC(kernel=tech)
    clf.fit(X_train,y_train)
    yhat=clf.predict(X_test)
    FS=f1_score(y_test,yhat,average='weighted')
    ACC=100*FS
    return ACC

def DT(independents,dependent,sp):
    from sklearn.tree import DecisionTreeClassifier
    X_train,X_test,y_train,y_test=train_test_split(df[independents],df[dependent],test_size=(1-(sp/100)),random_state=4)
    k=len(independents)
    mean_acc=np.zeros((k))
    std_acc=np.zeros((k))
    for n in range(1,k+1):
        Tree=DecisionTreeClassifier(criterion='entropy',max_depth=n)
        Tree.fit(X_train,y_train)
        Tree_Pred=Tree.predict(X_test)
        mean_acc[n-1]=metrics.accuracy_score(y_test,Tree_Pred)
        std_acc[n-1]=np.std(Tree_Pred==y_test)/np.sqrt(Tree_Pred.shape[0])
    for i in range(k):
        if mean_acc[i]==max(mean_acc):
            break
    K=i+1
    Tree=DecisionTreeClassifier(criterion='entropy',max_depth=K)
    Tree.fit(X_train,y_train)
    Tree_Pred=Tree.predict(X_test)
    FS=f1_score(y_test,Tree_Pred,average='weighted')
    ACC=100*FS
    return K,ACC

def LR(tech,independents,dependent,sp):
    from sklearn.linear_model import LogisticRegression
    X_train,X_test,y_train,y_test=train_test_split(df[independents],df[dependent],test_size=(1-(sp/100)),random_state=4)
    LR=LogisticRegression(C=0.01,solver=tech).fit(X_train,y_train)
    yhat=LR.predict(X_test)
    yhat_prob=LR.predict_proba(X_test)
    FS=f1_score(y_test,yhat,average='weighted')
    ACC=100*FS
    return ACC
   
global cols
global predictor
global df

#Flask
app = Flask(__name__)

#Loading Main Page (Upload CSV)
@app.route('/')
def upload_file_1():
   return render_template('upload.html')

#Loading Second page (Selecting predictor var)	
@app.route('/predictor', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      global df
      df=pd.read_csv(request.files['file'])
      df=df.dropna(axis=1)
      global cols
      cols=sorted(df)
      independents=cols
      data=[]
      for i,j in enumerate(cols,start=0):
         data.append({'col_name':j})
      #f.save(secure_filename(f.filename))
      return render_template('predictor.html', data=data)

#Loading Third page (Selecting to-be predicted var)
@app.route('/method', methods=['POST'])
def get_predictors():
   global cols
   global predictor
   predictor = request.form.getlist('predictor')
   predicted = list(set(cols) - set(predictor))
   data=[]
   for i,j in enumerate(predicted,start=0):
      data.append({'col_name':j})
   return render_template('predicted.html', data=data)

#Final Predicting
@app.route('/result', methods=['POST'])
def predict():
   opt=[]
   global predictor
   dependent=request.form.get('predicted')
   opt.append(request.form.get('opt'))
   sp=int(request.form.get('points'))
   opt.append(request.form.get('arg'))
   if opt[0]=='KNN':
       K,ACC=KNN(int(opt[1]),predictor,dependent,sp)
   elif opt[0]=='SVM':
       ACC=SVM(opt[1],predictor,dependent,sp)
   elif opt[0]=='DT':
       K,ACC=DT(predictor,dependent,sp)
   elif opt[0]=='LR':
       ACC=LR(opt[1],predictor,dependent,sp)
   
if __name__ == '__main__':
   app.run(debug = True)
