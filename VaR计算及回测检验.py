# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------
Spyder Editor          2020-05-06    



斯小伟  西南财经大学  金专1班 VaR的HS、HW、BRW方法预测及回顾测试
------------------------------------------------------------------
"""
import numpy as np
import pandas as pd
import os
import math
from matplotlib import pyplot as plt
os.chdir('C:\\Users\\GacidBru\\.spyder-py3')
path0 = os.getcwd()
data1 = pd.read_excel('IDXDVOLATILITY_C999A7581DA_(1).xls')
data3=data1.iloc[:,[2]]
list1=[]
data1=data1.iloc[:,[0,1,2]]
#---------------------------调整不同的置信度、滚动窗口天数、lamda--------------------------
confidence=0.99
window=100
lamda=0.997
#---------------------------------------------------------------------------------------
Brw=[]
HS=[]
HW=[]
sigma_new=data1.iloc[:,1]   #直接取出波动率数据--老师的论文方法
'''sigma_new=["0.015525"]
for i in range(1,data1.shape[0]):   #HULL WHITE的HW方法，用EWMA对波动率先进行更新,再对收益率更新排序，但检验结果不好
    sigma_new.append(np.sqrt(np.square(data1.iloc[i-1,1])*0.94+(1-0.94)*np.square(data1.iloc[i-1,2])))'''
for y in range(1,window+1):
    list1.append(np.power(lamda,int(y-1))*(1-lamda)/(1-np.power(lamda,window)))
for i in range(window+1,data1.shape[0]):
    datause = data1.iloc[i - window:i,2] 
    datause_list=np.array(datause)
    HS.append(np.percentile(datause,(1-confidence)*100))
    #HW方法，直接用波动率数据，对收益率进行更新，再排序
    m=sigma_new[i-1]
    z=np.array(sigma_new[i-window:i])
    datause2=datause_list*m/z
    HW.append(np.percentile(datause2,(1-confidence)*100))
    #BRW方法，先赋权重，再排序，再把排序的weight相加等于1-confidence，选出对应位置的return，就是var
    datause4=data1.iloc[i - window:i,2]
    datause4=datause4.to_frame()
    datause4.insert(0,'weight',list1)
    datause3=datause4.sort_values(by='Return',ascending=True)
    for j in range(2,window+2):
        if datause3.iloc[0:j-1,0].sum() > (1-confidence):
            Brw.append(datause3.iloc[j-2,1])
            break
#-------------------------------------VaR backtesting(把下方的Brw,HS,HW替换输入)--------------------------------
#UC  testing
r=0
for i in range(0,data1.shape[0]-window-1):
    if Brw[i]>=data1.iloc[i+window+1,2]:
        r=r+1
p=r/(data1.shape[0]-window-1) #r=T1 
LRUC = -2*math.log(((confidence**((data1.shape[0]-window-1)-r))*((1-confidence)**r))/(((1-p)**((data1.shape[0]-window-1)-r))*(p**r)),math.e)
# ind testing
qq=0
wq=0
qw=0
ww=0
for k in range(0,data1.shape[0]-window-2):
    if Brw[k]<data1.iloc[k+window+1,2] and Brw[k+1]<data1.iloc[k+window+2,2]:
        qq=qq+1
    if Brw[k]<data1.iloc[k+window+1,2] and Brw[k+1]>=data1.iloc[k+window+2,2]:
        wq=wq+1
    if Brw[k]>=data1.iloc[k+window+1,2] and Brw[k+1]<data1.iloc[k+window+2,2]:
        qw=qw+1
    if Brw[k]>=data1.iloc[k+window+1,2] and Brw[k+1]>=data1.iloc[k+window+2,2]:
        ww=ww+1
PQW=qw/(qq+wq)
PWW=ww/(wq+ww)
PP=(qw+ww)/(qq+qw+wq+ww)
LRIND=-2*(math.log(((1-PP)**(qq+wq))*(PP**(qw+ww)),math.e)-math.log(((1-PQW)**qq)*(PQW**qw)*((1-PWW)**wq)*(PWW**ww),math.e))
#CC
LRCC=LRIND+LRUC
    
print('\nLRTest     Value     chi-square(0.99) ',
      "-----------Bactesting-----------------\n"
      "LRUC     ",    round(LRUC,3)       ,"    6.635\n"
      "LRCC     ",     round(LRCC,3)       ,"    6.635\n"
      "LRIND    ",    round(LRIND,3)      ,"    9.210")
#---------------------------------------------作图---------------------------------------------
X = np.arange(len(HS)) 
#时间戳转化
xticklabel = data1.iloc[1+window:data1.shape[0],0].apply(lambda x:str(x.year) + '-' +str(x.month) + '-' +str(x.day))
xticklabel2=xticklabel.reset_index(drop = True)
def pick_arange(arange, num):                #这是一个保留首尾的arange函数，来自于https://www.zhongjianghua.com/att1tude/3236-2020-01.html
    if num > len(arange):
        #print("# num out of length, return arange:", end=" ")
        return arange
    else:
        output = np.array([], dtype=arange.dtype)
        seg = len(arange) / num
        for n in range(num):
            if int(seg * (n+1)) >= len(arange):
                output = np.append(output, arange[-1])
            else:
                output = np.append(output, arange[int(seg * n)])
        #print("# return new arange:", end=' ')
        return output            #索引重排
xticks = pick_arange(np.arange(0, len(HS)),10)     #xticks对应要展示的日期对应的索引
xticklabel3=[]
for i in range(0,len(xticks)):      #这一步是对要展示的横坐标索引进行日期定位
    gg=xticks[i]
    xticklabel3.append(xticklabel2[gg])   
#作图
plt.figure()
SP = plt.axes()
SP.plot(X,data1.iloc[1+window:data1.shape[0],2],color = 'mediumvioletred',label = 'Return')     #101:2436
SP.plot(X,HS,color = 'lightpink',label = 'VAR(HS)',ls='-.') 
SP.plot(X,HW,color = 'lightskyblue',label = 'VAR(HW)',ls='--')  
SP.plot(X,Brw,color = 'darkslateblue',label = 'VAR(BRW)',ls=':')   
SP.set_xticks(xticks)
SP.set_xticklabels(xticklabel3)
plt.xlabel(' Date')
plt.ylabel('VaR and Return')
plt.grid()
plt.legend()
