import mlflow
import pprint

def calculate_sum(x,y):
    sum=x*y
    return sum


if __name__=="__main__":
   # starting the mlflow sever
   with mlflow.start_run():
       x,y=30,90
       c=calculate_sum(x,y)
       # tracking the expirement with mlflow 
       mlflow.log_param("x",x)
       mlflow.log_param("y",y)
       mlflow.log_metric("c",c)
       print(f"the sum of two number is : {c}")