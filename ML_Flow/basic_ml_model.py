import pandas as pd
import numpy as  np
import os
import argparse

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,roc_auc_score
from sklearn.model_selection import train_test_split

def get_data(filename):
    try:
        data=pd.read_csv(filename,sep=';')
        return data
    except Exception as e:
        raise e


def evaluate_model(y_true,y_pred,pred_prob):
  ''' mae=mean_absolute_error(y_true,y_pred)
   mse=mean_squared_error(y_true,y_pred)
   rmse=np.sqrt(mean_squared_error(y_true,y_pred))
   r2=r2_score(y_true,y_pred)'''
  accuracy=accuracy_score(y_true,y_pred)
  roc_auc=roc_auc_score(y_true,pred_prob,multi_class="ovr")
  return accuracy,roc_auc


def main(n_estimators,max_depth):
    try:
     df=get_data("winequality-red.csv")
     train,test=train_test_split(df)
     x_train=train.drop(["quality"],axis=1)
     x_test=test.drop(["quality"],axis=1)
     y_train=train["quality"]
     y_test=test["quality"]
     
    #  lr=ElasticNet()
    #  lr.fit(x_train,y_train)
     with mlflow.start_run():
      rf=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
      rf.fit(x_train,y_train)
      pred=rf.predict(x_test)
      prob_pred=rf.predict_proba(x_test)
      accuracy,rocauc_score=evaluate_model(y_test,pred,prob_pred)
      mlflow.log_param("n_estimators",n_estimators)
      mlflow.log_param("max_depth",max_depth)
      mlflow.log_metric("accuracy",accuracy)
      mlflow.log_metric("roc_auc_score",rocauc_score)

      # mlflow model logging
      mlflow.sklearn.log_model(rf,"Random_Forest")

      print(accuracy,rocauc_score)


    except Exception as e:
       raise e



   


if __name__=="__main__":
   args=argparse.ArgumentParser()
   args.add_argument('--n_estimators','-n',default=100,type=int)
   args.add_argument('--max_depth','-m',default=5,type=int)
   parse_args=args.parse_args()
   try:
     main(n_estimators=parse_args.n_estimators,max_depth=parse_args.max_depth)
   except Exception as e:
      raise e
  
   