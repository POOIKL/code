#!/usr/bin/env python2.7
# -*- coding: utf-8 -*
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os

surf = cv2.xfeatures2d.SURF_create(300)
bf = cv2.BFMatcher_create()

# 匹配点去重，取最优
def Remove_duplication(TheList):
    # 返回测试图像去重后的点的索引 query_index 
    temp = [TheList[0]]
    query_index = [TheList[0][1]]
    count = 0
    for i in range(1, len(TheList)):
        if(TheList[i][0] == temp[count][0] and TheList[i][2] < temp[count][2]):
            temp[count] = TheList[i]
            query_index[count] = TheList[i][1]

        elif(TheList[i][0] != temp[count][0]):
            temp.append(TheList[i])
            query_index.append(TheList[i][1])
            count+=1
    return query_index



def ReadSingleNode(path):
    IL = []
    a = os.listdir(path)
    for i in a:
        IL.append(cv2.imread(path+i))    
    return IL



def GetSimilarity(ImgList1, ImgList2):
    ListSURF2 = {}
    ListSURF1 = {}

    for Index1 in range(len(ImgList1)):
        ListSURF1[Index1] = surf.detectAndCompute(ImgList1[Index1],None)[1]
    
    for Index2 in range(len(ImgList2)):
        ListSURF2[Index2] = surf.detectAndCompute(ImgList2[Index2],None)[1]


    p1List = []
    for i in range(len(ListSURF2)):
        p0List = []
        for j in range(len(ListSURF1)):
            matches = bf.match(ListSURF2[i], ListSURF1[j])
            GoodMatches = [(match.trainIdx, match.queryIdx, match.distance) for match in matches if match.distance < 0.3]
            QueryIndex = Remove_duplication(sorted(GoodMatches))
            p0 = float(len(QueryIndex)) / len(ListSURF2[i]) 
            p0List.append(p0)
        p1List.append(np.mean(p0List))
    
    return round(np.mean(p1List), 2)

