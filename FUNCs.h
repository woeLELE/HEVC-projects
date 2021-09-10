#pragma once
#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#include <crtdbg.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include "CxLibLinear.h"
using namespace std;
const int CTUS_LIMIT = 400;		// 可以使得CTU负类的个数至少为25个
const int TRAINING_SAMPLES2 = 1000;
const int LEN1 = 32;
const int LEN2 = 16;
const int PARTS1 = 4;
const int PARTS2 = 16;
enum { T2, P2, T3_2N, P3_2N, T3_N, P3_N };

// 以下两个函数用来根据传入的灰度图或残差图计算特征值
vector<double> calD(vector<vector<int>>& p);		// 计算局部方差以及平均方差
vector<double> SAGD(vector<vector<int>>& p);		// 计算方向复杂度以及方向复杂度方差

bool IsN(int a, int depth);	// 如果返回true，则为负类
int JudgeClass(int& cnt1, int& cnt0, int depth);	// 判断类别，因为某个深度可能进行正类预测（即使得正类查准率满足要求），也可能进行负类预测，还可能干脆不预测
CxLibLinear CreateModel(vector<vector<double>>& x, vector<double>& y, int& cnt1, int& cnt0, int label, int demand, char TorR, int D, int qp);	// 生成模型
vector<double> SetWeights(int cnt1, int cnt0, int label, int demand, char TorR, int D, int qp);	// 根据训练集中正类和负类的样本数确定权值比
CxLibLinear GetPredictor(double weight1, double weight0);		// 得到一个预测器
