import json
import numpy as np


with open("dataset.json",'r') as dataset_file:

    data=dataset_file.read()

jsondata=json.loads(data)

equation_dataset=[]

for data in jsondata:
    if('lEquations' in data.keys()):
        equation_dataset.append(data)

print(len(jsondata))
print(len(equation_dataset))

with open("dataset_with_equation.json",'w') as dataset_file:
    json.dump(equation_dataset,dataset_file)
