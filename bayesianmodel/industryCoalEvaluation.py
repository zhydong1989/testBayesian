# -*- coding: utf-8 -*-
"""
Created on Sat May 27 09:56:17 2017

@author: Yuyue
"""

import traceback
import logging
import pandas as pd
import numpy as np
import _pickle as cPickle
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.estimators import BdeuScore, K2Score, BicScore
from pgmpy.estimators import ExhaustiveSearch, HillClimbSearch
from pgmpy.estimators import ConstraintBasedEstimator
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
import copy
import networkx as nx
import os
import matplotlib.pyplot as plt

#os.chdir('C:/Users/Administrator/Documents/bayesianmodel')
curdir = os.path.abspath(os.path.dirname(__file__))

def bayesianModel(edges):
#edges: [(parent, child), (parent1, child1)]
    model = BayesianModel(edges)
    return model
def mleEstimator(model, data):
#最大似然估计
    try:
        model.fit(data, estimator = MaximumLikelihoodEstimator)
    except:
        logging.error(traceback.format_exc())
        return False, model
    return True, model
def bayesianEstimator(model, data):
#贝叶斯估计参数    
    try:
        model.fit(data, estimator = BayesianEstimator, prior_type = 'BDeu')
    except:
        logging.error(traceback.format_exc())
        return False, model
    return True, model
def InferenceVariable(model, variables, evidence):
#根据evidence查询variable的概率
#evidences: {var1:label, var2: label}
#variables: [var3, var4]
#返回值: 查询变量的概率字典
    result = {}
    try:
        infer = VariableElimination(model)
        inferresult = infer.query(variables = variables, evidence = evidence)
        for var in variables:
            result[var] = inferresult[var].values.astype('float32')
    except:
            logging.error(traceback.format_exc())
            return False, result
    return True, result
def Inference(model, variables, evidence, treshold):
#根据evidence查询variable的概率
#evidences: {var1:label, var2: label}
#variables: [var3, var4]
#treshold: 控制最大概率的阈值
#返回值: 最大概率取值/对应概率为字典
    result = {}
    try:
        infer = VariableElimination(model)
        inferresult = infer.query(variables = variables, evidence = evidence)
        for var in variables:
            tmpvalue = inferresult[var].values.astype('float32')
            print(tmpvalue)
            result[var] = {}
            if len(tmpvalue[tmpvalue == max(tmpvalue)]) == 1 and max(tmpvalue) > treshold:
                result[var] = {'value': np.argmax(tmpvalue), 'prob': max(tmpvalue)}
            else:
                result[var] = {'value': np.NaN, 'prob': np.NaN}
    except:
            print(traceback.format_exc())
            return False, result
    return True, result
  
def Predict_Direct(model, data_x):
    predict_data = data_x
    y_predict = model.predict(predict_data)
    return y_predict

def PredictEvaluation_Direct(y_predict, data_y):
    count = 0
    y_predict = y_predict.iloc[:, 0].values.tolist()
    data_y = data_y.iloc[:, 0].values.tolist()
    for i in range(len(y_predict)):
        if y_predict[i] == data_y[i]:
            count += 1
    return (count * 1.0)/len(y_predict)

def predictEvaluation(model, data_x, data_y):
#data_x: 测试集证据变量
#data_y: 测试集目标变量
#返回元组: (accuracy_counts,uncertain_counts,toal_counts)
    evidence = {}
    variables = str(data_y.columns.tolist()[0])
    predictresult = pd.DataFrame(np.zeros(np.shape(data_y)), 
                                 columns = [variables +'_predict'])
    predictresult.index = data_y.index
    jinducount = 0
    for index, row in data_x.iterrows():
        row = row.astype('int')
        tmpkey = data_x.columns.tolist()
        for item in tmpkey:
            evidence[item] = row[item]
        print(evidence)
        flag, result = Inference(model, [variables], evidence)
        print(flag)
        jinducount += 1
        print(jinducount)
        if flag:
            predictresult.ix[index,:] = result[variables]['value']
        else:
            predictresult.ix[index,:] = np.NaN
    data = pd.concat([data_y, predictresult], axis =1)
    totalsize = len(data_y)
    unknown = totalsize - len(predictresult.dropna())
    data = data.dropna()
    data = data.astype('float32')
    data = data.apply(lambda x: np.round(x))
    exact = len(data[data[variables] == data[variables+'_predict']])
    return (exact, unknown, totalsize)
    
def scoreStructureLearn(data, search='HillClimbSearch', scoring_method='BicScore'):
#基于score-search的结构学习
#search:HillClimbSearch, ExhaustiveSearch
#scoring_method: 'BicScore', K2Score, BdeuScore
    if scoring_method == 'BicScore':
        scoring_method_tmp = BicScore(data)
    elif scoring_method == 'K2Score':
        scoring_method_tmp = K2Score(data)
    elif scoring_method == 'BdeuScore':
        scoring_method_tmp = BdeuScore(data, equivalent_sample_size = 5)
    if search == 'HillClimbSearch':
        es = HillClimbSearch(data, scoring_method = scoring_method_tmp)
    else:
        es = ExhaustiveSearch(data, scoring_method = scoring_method_tmp)
    best_model = es.estimate()
    return best_model

def constraintStructureLearn(data, significance_level = 0.01):
#根据条件独立性约束构建贝叶斯网络
    est = ConstraintBasedEstimator(data)
    best_model = est.estimate(significance_level)
    return best_model
    
def evaluateRelation(model, variable):
#variable为model的节点
#返回各个父节点的取值下，variable的平均期望 dict, dict[item] 为一个dataframe    
    result = {}
    if not model.has_node(variable):
        return False, result
    #获得条件概率表
    variable_cpd = model.get_cpds(variable)
    #print(variable_cpd)
    #获得目标节点的取值范围
    leng = model.get_cardinality(variable)
    variable_quzhi = np.arange(leng).reshape((leng,1))
    #获得父亲节点的取值范围
    parent_list = model.get_parents(variable)
    tmp_parent_list = copy.deepcopy(parent_list)
    quzhi = {}
    for item in parent_list:
        quzhi[item] = model.get_cardinality(item)
    #获得某个父亲节点各种取值下，variable的平均期望
    for item in parent_list:
        tmp_parent_list.remove(item)
        #边缘化其他父亲节点，只保留item
        tmp = variable_cpd.marginalize(tmp_parent_list, False)
        #求item 节点的各个取值下， variable的期望值
        except_sum = np.sum(variable_quzhi*tmp.get_values(), axis = 0)
        result[item] = pd.DataFrame(except_sum, columns = ['Expection of the label of '+ 
                                    str(variable)], index = np.arange(quzhi[item]))
        tmp_parent_list = copy.deepcopy(parent_list)
    #print(result)
    return True, result

def saveModel(model):
    Graph = nx.DiGraph()
    Graph.add_edges_from(model.edges())
    plt.savefig(curdir+'/data/cixinstock_pkl'+'model.png')
    
def removeEdges(model, edges):
#输入：edges：[(parent1,child1),(parent2,child2)]
#notes: 返回的model 不再包含条件概率表,需要重新添加或者进行估计
    edges_tmp = model.edges()
    model.clear()
    new_edges = [edge for edge in edges_tmp if edge not in edges]
    model.add_edges_from(new_edges)
    return model

def addEdges(model, edges):
#输入：edges：[(parent1,child1),(parent2,child2)]
#notes: 返回的model 不再包含条件概率表,需要重新添加或者进行估计  
    edges_tmp = model.edges()
    model.clear()
    new_edges = list(set(edges_tmp).union(set(edges)))
    model.add_edges_from(new_edges)
    return model
    
def changeDirection(model, edges):
#输入：edges：[(parent1,child1),(parent2,child2)]. 如果 edge 与原有订单方向不同,删除。否则直接加入
#notes: 返回的model 不再包含条件概率表,需要重新添加或者进行估计  
    remove_edges_tmp =[]
    add_edges_tmp =[]
    old_edges = model.edges()
    for item in edges:
        tmp =(item[1],item[0])
        if tmp in old_edges:
            remove_edges_tmp.append(tmp)
        add_edges_tmp.append(item)
    modeltmp = removeEdges(model, remove_edges_tmp)
    modelresult = addEdges(modeltmp, add_edges_tmp)
    return modelresult

if __name__ == '__main__':
    label  = pd.read_csv('data/data_day_prepared_label.csv')
    length = 1000
    label_train = label[:length]
    label_test = label[length:]
    Execute = True
    while Execute:
        print('**************************************************************')
        print('1.scoreStructureLearn             2. constraintStructureLearn')
        print('3.parameterLearn                  (note:after 1/2')
        print('4.margianl distribution for special variable: (note:after 3')
        print('5.Analysis for attractive variables')
        print('6.bulid your own struction')
        print('7.why not guess')
        print('8.test')
        print('9.quit')
        print('**************************************************************')
        try:
            input_cmd = int(input("Welcome to bayesian network world! \
                                  Please choose from the above options:"))
            
            if input_cmd == 1:
                model = scoreStructureLearn(label_train)
                print(model.edges())
                cPickle.dump(model, open('model_scoreStructureLearn_stock.pkl', 'wb'))
            elif input_cmd == 2:
                model = constraintStructureLearn(label_train)
                print(model.edges())
                cPickle.dump(model, open('model_constraintLearn_stock.pkl', 'wb'))
            elif input_cmd == 3:
                model = cPickle.load(open('model_scoreStructureLearn_stock.pkl', 'rb'))
                flag, model = mleEstimator(model, label_train)
                if flag:
                    for cpd in model.get_cpds():
                        print(cpd)
                cPickle.dump(model, open('model_with_pL_stock.pkl', 'wb'))
            elif input_cmd == 4:
                model = cPickle.load(open('model_with_pL_stock.pkl', 'rb'))
                nodeslist = model.nodes()
                print(nodeslist)
                flag1 = True
                while flag1:
                    try:
                        variable = input("Please input your variable from the above variables:")
                        parent_list = model.get_parents(variable)
                        print(parent_list)
                        flag2 = True
                        if not parent_list:
                            print(model.get_cpds(variable))
                            flag2 = False
                        while flag2:
                            try:
                                parent_tmp = input("Please input the required variable's from the above\
                                               variables:")
                                parent_list.remove(parent_tmp)
                                variable_cpd = model.get_cpds(variable)
                                tmp = variable_cpd.marginalize(parent_list,False)
                                result = pd.DataFrame(tmp.get_values())
                                print(tmp)
                                flag2 = False
                            except:
                                print("single quote mark is required")
                        flag1 = False
                    except:
                        print("single quote mark is required")
            elif input_cmd == 5:
                model = cPickle.load(open('model_with_pL_stock.pkl', 'rb'))
                nodeslist = model.nodes()
                print(nodeslist)
                for item1 in nodeslist:
                    if 'Unnamed' in item1:
                        continue
                    flag, result = evaluateRelation(model, item1)
                    df_result = pd.DataFrame()
                    for key, item in result.items():
                        tmp = result[key]
                        tmp['parent'] =key
                        df_result =pd.concat([df_result, tmp], axis=0)
                    print(df_result)
            elif input_cmd == 6:
                model = cPickle.load(open('model_with_pL_stock.pkl','rb'))
                change_direction_list =[('51_sh_zxqyb_zongfaxingguben', 'revenue'),\
                                        ('51_sh_zxqyb_zongfaxingguben', 'profit'),\
                                        ('26_hs_a_zongshizhi', 'profit')]
                model = changeDirection(model, change_direction_list)
                print(model.edges())
                flag, model = mleEstimator(model, label_train)
                cPickle.dump(model, open('model_with_pL_stock_test.pkl', 'wb'))
                nodeslist = model.nodes()
                print(nodeslist)
                flag3= True
                while flag3:
                    try:
                        variable = input("Please input your variable from the above variables:")
                        parent_list = model.get_parents(variable)
                        print(parent_list)
                        flag4 = True
                        while flag4:
                            try:
                                parent_tmp = input("Please input the required variable's from the above\
                                               variables:")
                                parent_list.remove(parent_tmp)
                                variable_cpd = model.get_cpds(variable)
                                tmp = variable_cpd.marginalize(parent_list,False)
                                result = pd.DataFrame(tmp.get_values())
                                print(tmp)
                                flag4 = False
                            except:
                                print("single quote mark is required")
                        flag3 = False
                    except:
                        print("single quote mark is required")
                        print(traceback.format_exc())
            elif input_cmd == 7:
                model = cPickle.load(open('model_with_pL_stock.pkl','rb'))
                label_test_x = pd.DataFrame(label_test.ix[:, :-1])
                label_test_y = pd.DataFrame(label_test.ix[:, -1])
                predict_result = predictEvaluation(model, label_test_x, label_test_y)
                print(predict_result)
            elif input_cmd == 8 :
                model = cPickle.load(open('model_with_pL_stock_test.pkl','rb'))
                variable = ['revenue']
                nodeslist = model.nodes()
                columns_evidence = nodeslist
                delete_list = ['revenue', 'profit', '51_sh_zxqyb_zongfaxingguben']
                columns_evidence = list(set(columns_evidence)^set(delete_list))
                evidence = {}
                for item in columns_evidence:
                    evidence[item] = 2
                print(evidence)
                infer = VariableElimination(model)
                for i in range(4):
                    evidence['26_hs_a_zongshizhi'] = i
                    
                    inferresult = infer.query(variables=variable, evidence=evidence)
                    print(inferresult['revenue'])
            elif input_cmd == 9:
                Execute = False
                break
            else:
                print("please input integer limited in (1,9)")
        except:
            print("please input integer!")
      
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
