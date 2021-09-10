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
const int CTUS_LIMIT = 400;		// ����ʹ��CTU����ĸ�������Ϊ25��
const int TRAINING_SAMPLES2 = 1000;
const int LEN1 = 32;
const int LEN2 = 16;
const int PARTS1 = 4;
const int PARTS2 = 16;
enum { T2, P2, T3_2N, P3_2N, T3_N, P3_N };

// �������������������ݴ���ĻҶ�ͼ��в�ͼ��������ֵ
vector<double> calD(vector<vector<int>>& p);		// ����ֲ������Լ�ƽ������
vector<double> SAGD(vector<vector<int>>& p);		// ���㷽���Ӷ��Լ������Ӷȷ���

bool IsN(int a, int depth);	// �������true����Ϊ����
int JudgeClass(int& cnt1, int& cnt0, int depth);	// �ж������Ϊĳ����ȿ��ܽ�������Ԥ�⣨��ʹ�������׼������Ҫ�󣩣�Ҳ���ܽ��и���Ԥ�⣬�����ܸɴ಻Ԥ��
CxLibLinear CreateModel(vector<vector<double>>& x, vector<double>& y, int& cnt1, int& cnt0, int label, int demand, char TorR, int D, int qp);	// ����ģ��
vector<double> SetWeights(int cnt1, int cnt0, int label, int demand, char TorR, int D, int qp);	// ����ѵ����������͸����������ȷ��Ȩֵ��
CxLibLinear GetPredictor(double weight1, double weight0);		// �õ�һ��Ԥ����
