import os


n_estimators=[100,110,150,200,210]
max_depth=[5,10,15,25,20]

for n in n_estimators:
    for m in max_depth:
        os.system(f'python basic_ml_model.py -n{n} -m{m}')
                  