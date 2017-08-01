import json
import numpy as np
import re
import nltk

with open("dataset_with_equation.json",'r') as dataset_file:

    data=dataset_file.read()

jsondata=json.loads(data)

#extracting equations
equation_dataset=[]

for i in range(len(jsondata)):
    equation_dataset.append(jsondata[i]["lEquations"])

#removing whitespaces
equations_stripped=[]

for i in range(len(equation_dataset)):
    stripped=[]
    for j in range(len(equation_dataset[i])):
        stripped.append(equation_dataset[i][j].replace(' ',''))
    equations_stripped.append(stripped)


#changing equation to equation templates

equations_templatized=[]
for i in range(len(equations_stripped)):
    number_templatized=[]

    # var_dict={}
    # num_dict={}
    #
    # # for j in range(len(equations_stripped[i])):
    #
    #
    #
    #     var_search=re.findall(r"[A-Za-z_]+[0-9]?",equations_stripped[i][j])
    #     num_search = re.findall(r"\d*\.?\d+|\d+", equations_stripped[i][j])
    #
    #     if(var_search):
    #         for var in var_search:
    #             if var not in var_dict.keys():
    #                 var_dict[var] ="variable" + str(len(var_dict))
    #
    #     if (num_search):
    #         for num in num_search:
    #             if num not in num_dict.keys():
    #                 num_dict[num] = "number" + str(len(num_dict))
    #
    # for j in range(len(equations_stripped[i])):
    #
    #     var_sub = equations_stripped[i][j]
    #     for variable in var_dict.keys():
    #         var_sub = re.sub(variable, var_dict[variable], var_sub)
    #
    #     num_sub = var_sub
    #     for number in num_dict.keys():
    #         num_sub = re.sub(number, num_dict[number], num_sub)
    #
    #     number_templatized.append(num_sub)

    #replacing numbers with N and variables with X

    for j in range(len(equations_stripped[i])):

        var = re.sub(r"[A-Za-z_]+[0-9]?", "X",equations_stripped[i][j])

        num = re.sub(r"(\D[-+])?\d*\.\d+|\d+","N", var)

        number_templatized.append(num)

    equations_templatized.append(number_templatized)



#total number of unique equations (crude)
equation_list=[]
for i in range(len(equations_templatized)):
    for j in range(len(equations_templatized[i])):
        print((equation_dataset[i][j],equations_templatized[i][j]))

    if(not equation_list):              #if equation_list is empty, just append with count 1
        equation_list.append([equations_templatized[i], 1])
    elif(equations_templatized[i] not in (x[0] for x in equation_list)):  #if the equation isn't in the list
        equation_list.append([equations_templatized[i],1])                #append it to the list with count 1
    else:
        for k in range(len(equation_list)):
            if equation_list[k][0] == equations_templatized[i]:         #if equation exists on the list
                equation_list[k][1]+=1                                  #increase count by 1


#this sum prints total number of equations i.e equal to the dataset size
print(sum(x[1] for x in equation_list))

#sorting the equation list in descending order (equation with highest occurance first)

sorted_equation_list=sorted(equation_list,key=lambda x: x[1],reverse=True)


#equation type index assignment (equation with most occurances has lowest index)
equation_index=[]
for i in range(len(equations_templatized)):

    for j in range(len(equation_list)):
        if(equations_templatized[i]==sorted_equation_list[j][0]):
            equation_index.append(j)

#final list of dataset add equation class and the template equation into the dataset as well
final_dataset=[]
for i in range(len(equations_templatized)):
    jsondata[i]["equation_class"]=equation_index[i]
    jsondata[i]["equation_general"]=sorted_equation_list[equation_index[i]][0]


#write the dataset file
with open("dataset_with_equation_class.json",'w') as dataset_file:
    json.dump(jsondata,dataset_file)
