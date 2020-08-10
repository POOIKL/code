#!/usr/bin/env python2.7
# -*- coding: utf-8 -*
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import DirecTransform
import ImgMatch
import cv2
import networkx as nx


path = os.getcwd()
dirName = [dirnames for root,dirnames,filenames in os.walk(path)]
subDirList = list(filter(lambda x: re.match(r'(\d+)', x) != None, dirName[0]))
subdir = sorted(subDirList, key = lambda x:int(re.match(r'(\d+)', x).group()))


# 读取 txt 
def LoadTxt(load):
    Txtlist = []
    f = open(load + '/temp.txt')

    # 读取txt
    for line in f:
        Txtlist.append(line.strip())
    Txtlist[2] = float(Txtlist[2])
    f.close()    
    return Txtlist


# 读取图像集合
def LoadImg(load):
    ImgSet = []
    # 读取图像集合
    filenum = len(os.listdir(load + '/img'))
    for i in range(filenum):
        ImgSet.append(cv2.imread(load + '/img/' + str(i) + '.jpg'))

    return ImgSet  


def SaveFig(G, num, CurPos, lenDict):

    color_map = ['b' for i in range(len(lenDict))]
    color_map[CurPos -1] = 'r'

    plt.clf()
    nx.draw(g, nx.get_node_attributes(g, 'pos'), node_color=color_map, with_labels=True, node_size=300)
    plt.savefig('/home/mgn/mgn/projectpro/data-master/result/' + str(num) + '.jpg')



# 回转矩阵函数
def rotaMat(coordXY, DirVec, theta, distance):
    M = np.mat([[np.cos(theta * np.pi/180), -1 * np.sin(theta * np.pi/180)], \
        [np.sin(theta * np.pi/180), np.cos(theta * np.pi/180)]])

    ChangedDirVec = np.ravel(np.dot(M, DirVec))
    ChangedDirVec[0], ChangedDirVec[1] = int(ChangedDirVec[0]), int(ChangedDirVec[1])

    coordXY += ChangedDirVec * distance
    coordXY[0] = round(coordXY[0], 2)
    coordXY[1] = round(coordXY[1], 2)
    return coordXY.tolist(), ChangedDirVec 


# 计算距离
def calDistance(p1, p2):
    return np.sqrt(np.power((p2[0] - p1[0]), 2) + np.power((p2[1] - p1[1]), 2))




# 查找修正点
def SearchPoint(PointId, CurP, Dirpre, graph, CoorDict):
    dx = CurP[0] - CoorDict[PointId][0]
    dy = CurP[1] - CoorDict[PointId][1]

    if Dirpre[0] == 0 and Dirpre[1] == -1:
        if dx != 0:
            Flag = 'x'
        else:
            Flag = 'y'
    elif Dirpre[0] == 0 and Dirpre[1] == 1:
        if dx != 0:
            Flag = 'x'
        else:
            Flag = 'y'
    elif Dirpre[0] == 1 and Dirpre[1] == 0:
        if dy != 0:
            Flag = 'y'
        else:
            Flag = 'x'
    elif Dirpre[0] == -1 and Dirpre[1] == 0:
        if dy != 0:
            Flag = 'y'
        else:
            Flag = 'x'
    else:
        print('error')


    # 寻找同一直线上的点号
    SameLine = []
    if Flag == 'x':
        for i in range(1, len(CoorDict)+1):
            if CoorDict[i][0] == CurP[0]:
                SameLine.append((CoorDict[i][0],i))
    
    elif Flag == 'y':
        for i in range(1, len(CoorDict)+1):
            if CoorDict[i][1] == CurP[1]:
                SameLine.append((CoorDict[i][1],i))
    
    # 寻找中止条件
    ChangeCoord = []
    if SameLine != []:        
        if Dirpre[0] == 0 and Dirpre[1] == -1:
            SameLine = sorted(SameLine)
            ChangeCoord.append(SameLine[0][1])
            for i in range(1, len(SameLine)):
                n = len(nx.dijkstra_path(graph, source=SameLine[i-1][1], target=SameLine[i][1]))
                if n == 2:
                    ChangeCoord.append(SameLine[i][1])
                else:
                    break
        
        elif Dirpre[0] == 0 and Dirpre[1] == 1:
            SameLine = sorted(SameLine, reverse=True)
            ChangeCoord.append(SameLine[0][1])
            for i in range(1, len(SameLine)):
                n = len(nx.dijkstra_path(graph, source=SameLine[i-1], target=SameLine[i]))
                if n == 2:
                    ChangeCoord.append(SameLine[i][1])
                else:
                    break 

        elif Dirpre[0] == 1 and Dirpre[1] == 0:
            SameLine = sorted(SameLine)
            ChangeCoord.append(SameLine[0][1], reverse=True)
            for i in range(1, len(SameLine)):
                n = len(nx.dijkstra_path(graph, source=SameLine[i-1], target=SameLine[i]))
                if n == 2:
                    ChangeCoord.append(SameLine[i][1])
                else:
                    break 


        elif Dirpre[0] == - 1 and Dirpre[1] == 0:
            SameLine = sorted(SameLine)
            ChangeCoord.append(SameLine[0][1])
            for i in range(1, len(SameLine)):
                n = len(nx.dijkstra_path(graph, source=SameLine[i-1], target=SameLine[i]))
                if n == 2:
                    ChangeCoord.append(SameLine[i][1])
                else:
                    break 

    return ChangeCoord, Flag



def SearchNeighbors(CurPos, NodePre, CoordDict, g):
    neigh = []
    adj = []
    deltx = CurrentPosition[0] - CoordDict[NodePre][0]
    delty = CurrentPosition[1] - CoordDict[NodePre][1]

    adj = [i for i in g.neighbors(NodePre)]

    if deltx > 0:
        for i in adj:
            if (CoordDict[i][0] - CoordDict[NodePre][0]) > deltx:
                neigh.append(i)
    elif deltx < 0:
        for i in adj:
            if (CoordDict[i][0] - CoordDict[NodePre][0]) < deltx:
                neigh.append(i)
    elif delty > 0:
        for i in adj:
            if (CoordDict[i][1] - CoordDict[NodePre][1]) > delty:
                neigh.append(i)
    elif delty < 0:
        for i in adj:
            print(i)
            if (CoordDict[i][1] - CoordDict[NodePre][1]) < delty:
                neigh.append(i)
    return neigh




if __name__ == "__main__":
    
    NodeNow = 1
    NodePre = 1
    SaveImg = 1

    ReadNode = 1
    
    g = nx.Graph()
    ImgNodeDict = {}
    CoordNodeDict = {}

    for filenum in range(len(subdir)):
        Imgset = LoadImg('/home/mgn/mgn/projectpro/data-master/' + str(subdir[filenum]))
        txt = LoadTxt('/home/mgn/mgn/projectpro/data-master/' + str(subdir[filenum]))

        if ReadNode == 1:
            # 初始化坐标 方向
            CurrentPosition = [0, 0]
            DirVector = np.array([0, -1])
            
            # 字典记录图像集合 坐标
            ImgNodeDict[NodeNow] = Imgset
            CoordNodeDict[NodeNow] = CurrentPosition            

            # networkx 图更新
            g.add_node(NodeNow, pos=(CurrentPosition[0], CurrentPosition[1]))


            # 画图
            SaveFig(g, SaveImg, NodeNow, CoordNodeDict)
            
            NodePre = NodeNow
            #NodeNow += 1
            ReadNode += 1
            SaveImg += 1
        
        else:
            # 所在前一节点的方向向量
            DirPrevious = DirVector
            
            # 计算顺（逆）时针旋转角度
            RotAngle = DirecTransform.AngleTransform(DirVector, LoadTxt('/home/mgn/mgn/projectpro/data-master/' + str(subdir[filenum-1]))[1])

            # 计算当前坐标，方向向量进行变换
            CurrentPosition, DirVector = rotaMat(CurrentPosition, DirVector, RotAngle, txt[2])
            
            # 查找是否有图像比对的点
            similarset = []
            for i in range(1, len(CoordNodeDict)+1):
                if calDistance(CurrentPosition, CoordNodeDict[i]) < 7:
                    simiValue = ImgMatch.GetSimilarity(Imgset, ImgNodeDict[i])
                    if simiValue > 0.2:
                        similarset.append((simiValue, i))
            similarset = sorted(similarset, reverse=True)

            
            # 地图修正
            if similarset != []:
                ModifyPointSet, RotFlag = SearchPoint(similarset[0][1], CurrentPosition, DirPrevious, g, CoordNodeDict)
            
                if ModifyPointSet != []:
                    if RotFlag == 'x':
                        #NodeNow = similarset[0][1]
                        detx = CoordNodeDict[similarset[0][1]][0] - CurrentPosition[0]
                        g.add_edges_from([(ModifyPointSet[0], similarset[0][1])])

                        # 整体修正
                        for i in ModifyPointSet:
                            CoordNodeDict[i][0] += detx
                            g.add_node(i, pos=(CoordNodeDict[i][0], CoordNodeDict[i][1]))
                        
                        
                        CurrentPosition = CoordNodeDict[similarset[0][1]]
                        SaveFig(g, SaveImg, similarset[0][1], CoordNodeDict)
                        NodePre = similarset[0][1]
                        SaveImg += 1

                
                    elif RotFlag == 'y':
                        #NodeNow = similarset[0][1]
                        dety = CoordNodeDict[similarset[0][1]][1] - CurrentPosition[1]
                        g.add_edges_from([(ModifyPointSet[0], similarset[0][1])])

                        # 整体修正
                        for i in ModifyPointSet:
                            CoordNodeDict[i][1] += dety
                            g.add_node(i, pos=(CoordNodeDict[i][0], CoordNodeDict[i][1]))
                        
                        
                        CurrentPosition = CoordNodeDict[similarset[0][1]]
                        SaveFig(g, SaveImg, similarset[0][1], CoordNodeDict)
                        NodePre = similarset[0][1]
                        SaveImg += 1

                elif ModifyPointSet == []:
                    #NodeNow = similarset[0][1]
                    g.add_edges_from([(NodePre, similarset[0][1])])
                    
                    CurrentPosition = CoordNodeDict[similarset[0][1]]
                    SaveFig(g, SaveImg, similarset[0][1], CoordNodeDict)
                    NodePre = similarset[0][1]
                    SaveImg += 1


            else:
                neighbor = []
                NodeNow += 1
                CoordNodeDict[NodeNow] = CurrentPosition
                ImgNodeDict[NodeNow] = Imgset

                neighbor = SearchNeighbors(CurrentPosition, NodePre, CoordNodeDict, g)

                if neighbor != []:                    
                    g.add_node(NodeNow, pos=(CurrentPosition[0], CurrentPosition[1]))
                    g.remove_edge(neighbor[0], NodePre)
                    g.add_edges_from([(NodeNow, NodePre)])
                    g.add_edges_from([(NodeNow, neighbor[0])])

                else:
                    g.add_node(NodeNow, pos=(CurrentPosition[0], CurrentPosition[1]))
                    g.add_edges_from([(NodeNow, NodePre)])
                
                NodePre = NodeNow
                SaveFig(g, SaveImg, NodeNow, CoordNodeDict)
                SaveImg += 1