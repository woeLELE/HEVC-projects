/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2015, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

 /** \file     TEncCu.cpp
	 \brief    Coding Unit (CU) encoder class
 */

#include <stdio.h>
#include "TEncTop.h"
#include "TEncCu.h"
#include "TEncAnalyze.h"
#include "TLibCommon/Debug.h"

#include <cmath>
#include <set>
#include <algorithm>
#include <deque>
#include "FUNCs.h"
using namespace std;
int num = 0;
//#define PARTS1 4
//#define PARTS2 16
//const string P("G");
int strides[] = { 16, 8, 4, 2, 1 };
//int Size[] = { 64, 32, 16, 8 };
//int pp[64][64];
int flag[3] = { -1, -1, -1 };
bool data0 = false;
//bool data1 = false;
bool superSplit0 = false;
bool superCease2 = false;
vector<vector<double>> train_xT0;
vector<vector<double>> train_xT1;
vector<vector<double>> train_xR1;
vector<vector<double>> train_xT2;
vector<double> train_yT0;
vector<double> train_yT1;
vector<double> train_yR1;
vector<double> train_yT2;
int cnt_train[3][2] = { 0 };	// 各个深度下正类和负类的样本数
int cnt_traRMD1 = 0;
int cnt_traRMD0 = 0;
vector<vector<double>> train2_x;	// 二类data2
vector<double>	train2_y;

int cnt_tempT0_1 = 0;
int cnt_tempT0_0 = 0;
int cnt_tempT1_1 = 0;
int cnt_tempT1_0 = 0;
int cnt_tempT2_1 = 0;
int cnt_tempT2_0 = 0;
int cnt_tempR1 = 0;
int cnt_tempR0 = 0;
vector<CxLibLinear> models;
string model_path0 = "linear_model0.txt";
CxLibLinear tlinear0;
string model_path1 = "linear_model1.txt";
CxLibLinear tlinear1;
string model_path2 = "linear_model2.txt";
CxLibLinear tlinear2_90;
string model_path2L = "linear_model2L.txt";
CxLibLinear tlinear2_60;
int value;
double prob_est;

int preValue0 = -1;
//vector<int> preValue1;
//vector<int> preValue2;
ofstream out0("输出0.txt");
ofstream out1("输出1.txt");
ofstream out2("输出2.txt");
//ofstream out3("输出3.txt");

ofstream tout("调试输出.txt");
bool isComplete = true;				// 判断CTU是否完整
vector<double> tpSample;
vector<vector<double>> sample1T;
vector<double> sample0;				// 深度0样本
vector<vector<double>> sample2;
int r0 = -1;
vector<int> r1;
vector<int> r2;
//vector<int> r3;
//vector<int> r4;
double label;
float a[4][4];

float qp;
bool judged_qp = false;
vector<double> ND;            // 方差
vector<double> SAG;           // 方向复杂度及方向复杂度方差

//double rsdF[2];      // 残差偏度和峰度
//double* rsdSF;		// 残差子复杂度方差
//vector<double> rsdND;		// 残差局部方差和全局方差
//vector<double> rsdSAG;		// 残差方向复杂度和方向复杂度方差
//double rsdAve;		// 残差均值
//double rsdU[2];
//double rsdD;			// 残差方差
//double rsdT;			// 残差和正态分布拟合度
//double RDCost;		// 残差的RDCost
int ctuNum = 0;     //  用来记录CTU的个数
int m0;		// 一行中CTU的个数
int n0;		// 一列中CTU的个数
int ctuIndex = 0;
//int px = 0, py = 0;
//int indexD1 = 0;
//int indexD2 = 0;
//int indexD3 = 0;
bool gotSize = false;
//vector<bool> used1(4, false);
//vector<bool> used2(16, false);
//vector<bool> used3(64, false);
int pre0 = -1;		// -1代表不确定正类还是负类，1代表认为是正类，0代表认为是负类
int pre1 = -1;
int pre1_1 = -1;
int pre2 = -1;
int pre1_2 = -1;
bool split = false;
bool cease = false;
bool doRMD = false;
int cnt_tempRMD2_1 = 0;
int cnt_tempRMD2_0 = 0;
int fr = 0;
bool sflag = false;
bool sInfo0 = false;
vector<bool> sInfo1;
vector<bool> sInfo2;
//! \ingroup TLibEncoder
//! \{

// ====================================================================================================================
// Constructor / destructor / create / destroy
// ====================================================================================================================

/**
 \param    uhTotalDepth  total number of allowable depth
 \param    uiMaxWidth    largest CU width
 \param    uiMaxHeight   largest CU height
 \param    chromaFormat  chroma format
 */
Void TEncCu::create(UChar uhTotalDepth, UInt uiMaxWidth, UInt uiMaxHeight, ChromaFormat chromaFormat)
{
	Int i;

	m_uhTotalDepth = uhTotalDepth + 1;
	m_ppcBestCU = new TComDataCU * [m_uhTotalDepth - 1];
	m_ppcTempCU = new TComDataCU * [m_uhTotalDepth - 1];

	m_ppcPredYuvBest = new TComYuv * [m_uhTotalDepth - 1];
	m_ppcResiYuvBest = new TComYuv * [m_uhTotalDepth - 1];
	m_ppcRecoYuvBest = new TComYuv * [m_uhTotalDepth - 1];
	m_ppcPredYuvTemp = new TComYuv * [m_uhTotalDepth - 1];
	m_ppcResiYuvTemp = new TComYuv * [m_uhTotalDepth - 1];
	m_ppcRecoYuvTemp = new TComYuv * [m_uhTotalDepth - 1];
	m_ppcOrigYuv = new TComYuv * [m_uhTotalDepth - 1];

	UInt uiNumPartitions;
	for (i = 0; i < m_uhTotalDepth - 1; i++)
	{
		uiNumPartitions = 1 << ((m_uhTotalDepth - i - 1) << 1);
		UInt uiWidth = uiMaxWidth >> i;
		UInt uiHeight = uiMaxHeight >> i;

		m_ppcBestCU[i] = new TComDataCU; m_ppcBestCU[i]->create(chromaFormat, uiNumPartitions, uiWidth, uiHeight, false, uiMaxWidth >> (m_uhTotalDepth - 1));
		m_ppcTempCU[i] = new TComDataCU; m_ppcTempCU[i]->create(chromaFormat, uiNumPartitions, uiWidth, uiHeight, false, uiMaxWidth >> (m_uhTotalDepth - 1));

		m_ppcPredYuvBest[i] = new TComYuv; m_ppcPredYuvBest[i]->create(uiWidth, uiHeight, chromaFormat);
		m_ppcResiYuvBest[i] = new TComYuv; m_ppcResiYuvBest[i]->create(uiWidth, uiHeight, chromaFormat);
		m_ppcRecoYuvBest[i] = new TComYuv; m_ppcRecoYuvBest[i]->create(uiWidth, uiHeight, chromaFormat);

		m_ppcPredYuvTemp[i] = new TComYuv; m_ppcPredYuvTemp[i]->create(uiWidth, uiHeight, chromaFormat);
		m_ppcResiYuvTemp[i] = new TComYuv; m_ppcResiYuvTemp[i]->create(uiWidth, uiHeight, chromaFormat);
		m_ppcRecoYuvTemp[i] = new TComYuv; m_ppcRecoYuvTemp[i]->create(uiWidth, uiHeight, chromaFormat);

		m_ppcOrigYuv[i] = new TComYuv; m_ppcOrigYuv[i]->create(uiWidth, uiHeight, chromaFormat);
	}

	m_bEncodeDQP = false;
	m_stillToCodeChromaQpOffsetFlag = false;
	m_cuChromaQpOffsetIdxPlus1 = 0;
	m_bFastDeltaQP = false;

	// initialize partition order.
	UInt* piTmp = &g_auiZscanToRaster[0];
	initZscanToRaster(m_uhTotalDepth, 1, 0, piTmp);
	initRasterToZscan(uiMaxWidth, uiMaxHeight, m_uhTotalDepth);

	// initialize conversion matrix from partition index to pel
	initRasterToPelXY(uiMaxWidth, uiMaxHeight, m_uhTotalDepth);
}

Void TEncCu::destroy()
{
	Int i;

	for (i = 0; i < m_uhTotalDepth - 1; i++)
	{
		if (m_ppcBestCU[i])
		{
			m_ppcBestCU[i]->destroy();      delete m_ppcBestCU[i];      m_ppcBestCU[i] = NULL;
		}
		if (m_ppcTempCU[i])
		{
			m_ppcTempCU[i]->destroy();      delete m_ppcTempCU[i];      m_ppcTempCU[i] = NULL;
		}
		if (m_ppcPredYuvBest[i])
		{
			m_ppcPredYuvBest[i]->destroy(); delete m_ppcPredYuvBest[i]; m_ppcPredYuvBest[i] = NULL;
		}
		if (m_ppcResiYuvBest[i])
		{
			m_ppcResiYuvBest[i]->destroy(); delete m_ppcResiYuvBest[i]; m_ppcResiYuvBest[i] = NULL;
		}
		if (m_ppcRecoYuvBest[i])
		{
			m_ppcRecoYuvBest[i]->destroy(); delete m_ppcRecoYuvBest[i]; m_ppcRecoYuvBest[i] = NULL;
		}
		if (m_ppcPredYuvTemp[i])
		{
			m_ppcPredYuvTemp[i]->destroy(); delete m_ppcPredYuvTemp[i]; m_ppcPredYuvTemp[i] = NULL;
		}
		if (m_ppcResiYuvTemp[i])
		{
			m_ppcResiYuvTemp[i]->destroy(); delete m_ppcResiYuvTemp[i]; m_ppcResiYuvTemp[i] = NULL;
		}
		if (m_ppcRecoYuvTemp[i])
		{
			m_ppcRecoYuvTemp[i]->destroy(); delete m_ppcRecoYuvTemp[i]; m_ppcRecoYuvTemp[i] = NULL;
		}
		if (m_ppcOrigYuv[i])
		{
			m_ppcOrigYuv[i]->destroy();     delete m_ppcOrigYuv[i];     m_ppcOrigYuv[i] = NULL;
		}
	}
	if (m_ppcBestCU)
	{
		delete[] m_ppcBestCU;
		m_ppcBestCU = NULL;
	}
	if (m_ppcTempCU)
	{
		delete[] m_ppcTempCU;
		m_ppcTempCU = NULL;
	}

	if (m_ppcPredYuvBest)
	{
		delete[] m_ppcPredYuvBest;
		m_ppcPredYuvBest = NULL;
	}
	if (m_ppcResiYuvBest)
	{
		delete[] m_ppcResiYuvBest;
		m_ppcResiYuvBest = NULL;
	}
	if (m_ppcRecoYuvBest)
	{
		delete[] m_ppcRecoYuvBest;
		m_ppcRecoYuvBest = NULL;
	}
	if (m_ppcPredYuvTemp)
	{
		delete[] m_ppcPredYuvTemp;
		m_ppcPredYuvTemp = NULL;
	}
	if (m_ppcResiYuvTemp)
	{
		delete[] m_ppcResiYuvTemp;
		m_ppcResiYuvTemp = NULL;
	}
	if (m_ppcRecoYuvTemp)
	{
		delete[] m_ppcRecoYuvTemp;
		m_ppcRecoYuvTemp = NULL;
	}
	if (m_ppcOrigYuv)
	{
		delete[] m_ppcOrigYuv;
		m_ppcOrigYuv = NULL;
	}
}

/** \param    pcEncTop      pointer of encoder class
 */
Void TEncCu::init(TEncTop* pcEncTop)
{
	m_pcEncCfg = pcEncTop;
	m_pcPredSearch = pcEncTop->getPredSearch();
	m_pcTrQuant = pcEncTop->getTrQuant();
	m_pcRdCost = pcEncTop->getRdCost();

	m_pcEntropyCoder = pcEncTop->getEntropyCoder();
	m_pcBinCABAC = pcEncTop->getBinCABAC();

	m_pppcRDSbacCoder = pcEncTop->getRDSbacCoder();
	m_pcRDGoOnSbacCoder = pcEncTop->getRDGoOnSbacCoder();

	m_pcRateCtrl = pcEncTop->getRateCtrl();
}

// ====================================================================================================================
// Public member functions
// ====================================================================================================================

/**
 \param  pCtu pointer of CU data class
 */
Void TEncCu::compressCtu(TComDataCU* pCtu)
{
	sInfo0 = false;
	sInfo1.clear();
	sInfo2.clear();
	isComplete = true;
	if (!data0)
	{
		sample1T.clear();
		sample2.clear();
	}
	m_ppcBestCU[0]->initCtu(pCtu->getPic(), pCtu->getCtuRsAddr());	// picture class pointer
	m_ppcTempCU[0]->initCtu(pCtu->getPic(), pCtu->getCtuRsAddr());	// CTU (also known as LCU) address in a slice (Raster-scan address, as opposed to tile-scan/encoding order).
	if (!gotSize)
	{
		m0 = m_ppcBestCU[0]->getPic()->getFrameWidthInCtus();
		n0 = m_ppcBestCU[0]->getPic()->getFrameHeightInCtus();
		gotSize = true;
	}
	//px = ctuIndex / m0;
	//py = ctuIndex % m0;
	// analysis of CU
	DEBUG_STRING_NEW(sDebug)

		xCompressCU(m_ppcBestCU[0], m_ppcTempCU[0], 0 DEBUG_STRING_PASS_INTO(sDebug));
	DEBUG_STRING_OUTPUT(std::cout, sDebug)
		int max = 0, i, j, x, y, d[16][16], m = -1, n, k = 0;
	int iCount = 0;
	int iWidthInPart = MAX_CU_SIZE >> 2;
	for (int i = 0; i < pCtu->getTotalNumPart(); i++)
	{
		if ((iCount & (iWidthInPart - 1)) == 0)
		{
			m++;
			n = 0;
		}
		d[m][n++] = pCtu->getDepth(g_auiRasterToZscan[i]);
		iCount++;
	}
	r1.clear();
	r2.clear();
	// 如果当前CTU是完整的并且还没有生成纹理模型
	if (isComplete && !data0)
	{
		// 训练集生成完毕但还没有生成模型
		if (ctuNum >= CTUS_LIMIT && fr > 0)
		{
			for (int depth = 0; depth <= 2; depth++)
				flag[depth] = JudgeClass(cnt_train[depth][1], cnt_train[depth][0], depth);
			if (cnt_train[0][1] > (15 * cnt_train[0][0]))
			{
				superSplit0 = true;
				tout << "superSplit0" << endl;
			}	 // 特殊情况
			if (cnt_train[2][0] > (15 * cnt_train[2][1]))
			{
				superCease2 = true;
				tout << "superCease2" << endl;
			}
			tout << "该序列各个深度训练集的类别：" << endl;
			tout << flag[0] << " -- " << flag[1] << " -- " << flag[2] << endl;

			// 生成纹理模型
			if (flag[0] == 1)
			{
				CxLibLinear tm0 = CreateModel(train_xT0, train_yT0, cnt_train[0][1], cnt_train[0][0], 1, 90, 'T', 0, qp);
				tm0.save_linear_model(model_path0);
				tlinear0.load_linear_model(model_path0);
			}
			if (flag[1] == 1)
			{
				CxLibLinear tm1 = CreateModel(train_xT1, train_yT1, cnt_train[1][1], cnt_train[1][0], 1, 90, 'T', 1, qp);
				tm1.save_linear_model(model_path1);
				tlinear1.load_linear_model(model_path1);
			}
			else if (flag[1] == 2)
			{
				CxLibLinear tm1 = CreateModel(train_xT1, train_yT1, cnt_train[1][1], cnt_train[1][0], 0, 90, 'T', 1, qp);
				tm1.save_linear_model(model_path1);
				tlinear1.load_linear_model(model_path1);
			}
			if (flag[2] == 1)
			{
				CxLibLinear tm2 = CreateModel(train_xT2, train_yT2, cnt_train[2][1], cnt_train[2][0], 0, 90, 'T', 2, qp);
				tm2.save_linear_model(model_path2);
				tlinear2_90.load_linear_model(model_path2);

				CxLibLinear tm3 = CreateModel(train_xT2, train_yT2, cnt_train[2][1], cnt_train[2][0], 0, 60, 'T', 2, qp);
				tm3.save_linear_model(model_path2L);
				tlinear2_60.load_linear_model(model_path2L);
			}
			else if (flag[2] == 2)
			{
				CxLibLinear tm3 = CreateModel(train_xT2, train_yT2, cnt_train[2][1], cnt_train[2][0], 0, 60, 'T', 2, qp);
				tm3.save_linear_model(model_path2L);
				tlinear2_60.load_linear_model(model_path2L);
			}
			data0 = true;
		}
		// 继续补充训练集
		if (!data0)
		{
			ctuNum++;
			if (ctuNum % 500 == 0)
				tout << ctuNum << endl;

			for (int depth = 0; depth < 3; depth++)		// 得到不同大小CU的深度
			{
				int stride = strides[depth];
				for (i = 0; i < 16; i = i + stride)
				{
					for (j = 0; j < 16; j = j + stride)
					{
						max = 0;
						for (m = i; m < i + stride; m++)
						{
							for (n = j; n < j + stride; n++)
							{
								if (max < d[m][n])
									max = d[m][n];
								if (max == 3)
									break;
							}
							if (max == 3)
								break;
						}
						if (depth == 0)
							r0 = max;
						else if (depth == 1)
							r1.push_back(max);
						else if (depth == 2)
							r2.push_back(max);
					}
				}
			}
			out0 << (bool)(r0 > 0) << " : " << sInfo0 << endl;
			for (auto d : r1)
				out1 << (bool)(d > 1) << " ";
			out1 << endl;
			for (auto f : sInfo1)
				out1 << f << " ";
			out1 << endl;
			out1 << "---------------------------------" << endl;
			for (auto d : r2)
				out2 << (bool)(d > 2) << " ";
			out2 << endl;
			for (auto f : sInfo2)
				out2 << f << " ";
			out2 << endl;
			out2 << "---------------------------------" << endl;
			// 生成深度0纹理训练集
			if (!(sample0[2] < -10 && sample0[3] < -10 && sample0[4] < -10 && sample0[1] < -10))
			{
				bool flag = IsN(r0, 0);
				if (!flag)
					cnt_train[0][1]++;
				else
					cnt_train[0][0]++;
				if (flag && cnt_tempT0_1 - cnt_tempT0_0 <= 1 && cnt_tempT0_1 - cnt_tempT0_0 >= 0)	// 用以实现均等化
				{
					cnt_tempT0_0++;
					train_yT0.push_back(0);
					train_xT0.push_back(sample0);
				}
				else if (!flag && cnt_tempT0_0 - cnt_tempT0_1 <= 1 && cnt_tempT0_0 - cnt_tempT0_1 >= 0)
				{
					cnt_tempT0_1++;
					train_yT0.push_back(1);
					train_xT0.push_back(sample0);
				}
			}
			// 生成深度1纹理训练集
			if (sample1T.size() == PARTS1)
			{
				for (i = 0; i < PARTS1; i++)
				{
					if (sample1T[i][2] < -10)
						continue;
					bool flag = IsN(r1[i], 1);
					if (!flag)
						cnt_train[1][1]++;
					else
						cnt_train[1][0]++;
					if (flag && cnt_tempT1_1 - cnt_tempT1_0 <= 1 && cnt_tempT1_1 - cnt_tempT1_0 >= 0)
					{
						cnt_tempT1_0++;
						train_yT1.push_back(0);
					}
					else if (!flag && cnt_tempT1_0 - cnt_tempT1_1 <= 1 && cnt_tempT1_0 - cnt_tempT1_1 >= 0)
					{
						cnt_tempT1_1++;
						train_yT1.push_back(1);
					}
					else
						continue;
					train_xT1.push_back(sample1T[i]);
				}

			}
			// 生成深度2纹理训练集
			if (sample2.size() == PARTS2 && fr == 0)
			{
				for (i = 0; i < PARTS2; i++)
				{
					if (sample2[i][2] < -10)
						continue;
					bool flag = IsN(r2[i], 1);
					if (!flag)
						cnt_train[2][1]++;
					else
						cnt_train[2][0]++;
					if (flag && cnt_tempT2_1 - cnt_tempT2_0 <= 1 && cnt_tempT2_1 - cnt_tempT2_0 >= 0)
					{
						cnt_tempT2_0++;
						train_yT2.push_back(0);
					}
					else if (!flag && cnt_tempT2_0 - cnt_tempT2_1 <= 1 && cnt_tempT2_0 - cnt_tempT2_1 >= 0)
					{
						cnt_tempT2_1++;
						train_yT2.push_back(1);
					}
					else
						continue;
					train_xT2.push_back(sample2[i]);
				}
			}
		}
	}
	ctuIndex++;
	if (ctuIndex == m0 * n0)
	{
		ctuIndex = 0;
		fr++;
	}
#if ADAPTIVE_QP_SELECTION
	if (m_pcEncCfg->getUseAdaptQpSelect())
	{
		/*
		TComSlice*  getSlice  ( ) { return m_pcSlice; }
		---------------------------------
		SliceType   getSliceType() const   { return m_eSliceType; }

		enum SliceType
		{
		  B_SLICE               = 0,
		  P_SLICE               = 1,
		  I_SLICE               = 2,
		  NUMBER_OF_SLICE_TYPES = 3
		};
		*/
		// getSlice����TComSlice���ָ�룬�����е�getSliceType����SliceType���ö��
		if (pCtu->getSlice()->getSliceType() != I_SLICE)
		{
			xCtuCollectARLStats(pCtu);
		}
	}
#endif
}
/** \param  pCtu  pointer of CU data class
 */
Void TEncCu::encodeCtu(TComDataCU* pCtu)
{
	if (pCtu->getSlice()->getPPS()->getUseDQP())
	{
		setdQPFlag(true);
	}

	if (pCtu->getSlice()->getUseChromaQpAdj())
	{
		setCodeChromaQpAdjFlag(true);
	}

	// Encode CU data
	xEncodeCU(pCtu, 0, 0);
}

// ====================================================================================================================
// Protected member functions
// ====================================================================================================================
//! Derive small set of test modes for AMP encoder speed-up
#if AMP_ENC_SPEEDUP
#if AMP_MRG
Void TEncCu::deriveTestModeAMP(TComDataCU* pcBestCU, PartSize eParentPartSize, Bool& bTestAMP_Hor, Bool& bTestAMP_Ver, Bool& bTestMergeAMP_Hor, Bool& bTestMergeAMP_Ver)
#else
Void TEncCu::deriveTestModeAMP(TComDataCU* pcBestCU, PartSize eParentPartSize, Bool& bTestAMP_Hor, Bool& bTestAMP_Ver)
#endif
{
	if (pcBestCU->getPartitionSize(0) == SIZE_2NxN)
	{
		bTestAMP_Hor = true;
	}
	else if (pcBestCU->getPartitionSize(0) == SIZE_Nx2N)
	{
		bTestAMP_Ver = true;
	}
	else if (pcBestCU->getPartitionSize(0) == SIZE_2Nx2N && pcBestCU->getMergeFlag(0) == false && pcBestCU->isSkipped(0) == false)
	{
		bTestAMP_Hor = true;
		bTestAMP_Ver = true;

	}

#if AMP_MRG
	//! Utilizing the partition size of parent PU
	if (eParentPartSize >= SIZE_2NxnU && eParentPartSize <= SIZE_nRx2N)
	{
		bTestMergeAMP_Hor = true;
		bTestMergeAMP_Ver = true;
	}

	if (eParentPartSize == NUMBER_OF_PART_SIZES) //! if parent is intra
	{
		if (pcBestCU->getPartitionSize(0) == SIZE_2NxN)
		{
			bTestMergeAMP_Hor = true;
		}
		else if (pcBestCU->getPartitionSize(0) == SIZE_Nx2N)
		{
			bTestMergeAMP_Ver = true;
		}
	}

	if (pcBestCU->getPartitionSize(0) == SIZE_2Nx2N && pcBestCU->isSkipped(0) == false)
	{
		bTestMergeAMP_Hor = true;
		bTestMergeAMP_Ver = true;
	}

	if (pcBestCU->getWidth(0) == 64)
	{
		bTestAMP_Hor = false;
		bTestAMP_Ver = false;
	}
#else
	//! Utilizing the partition size of parent PU
	if (eParentPartSize >= SIZE_2NxnU && eParentPartSize <= SIZE_nRx2N)
	{
		bTestAMP_Hor = true;
		bTestAMP_Ver = true;
	}

	if (eParentPartSize == SIZE_2Nx2N)
	{
		bTestAMP_Hor = false;
		bTestAMP_Ver = false;
	}
#endif
}
#endif


// ====================================================================================================================
// Protected member functions
// ====================================================================================================================
/** Compress a CU block recursively with enabling sub-CTU-level delta QP
 *  - for loop of QP value to compress the current CU with all possible QP
*/
/*
  m_ppcBestCU[0]->initCtu( pCtu->getPic(), pCtu->getCtuRsAddr() );	// picture class pointer
  m_ppcTempCU[0]->initCtu( pCtu->getPic(), pCtu->getCtuRsAddr() );	// CTU (also known as LCU) address in a slice (Raster-scan address, as opposed to tile-scan/encoding order).

  // analysis of CU
  DEBUG_STRING_NEW(sDebug)

  xCompressCU( m_ppcBestCU[0], m_ppcTempCU[0], 0 DEBUG_STRING_PASS_INTO(sDebug) );
  */
  /*
  enum PartSize
  {
	SIZE_2Nx2N           = 0,           ///< symmetric motion partition,  2Nx2N
	SIZE_2NxN            = 1,           ///< symmetric motion partition,  2Nx N
	SIZE_Nx2N            = 2,           ///< symmetric motion partition,   Nx2N
	SIZE_NxN             = 3,           ///< symmetric motion partition,   Nx N
	SIZE_2NxnU           = 4,           ///< asymmetric motion partition, 2Nx( N/2) + 2Nx(3N/2)
	SIZE_2NxnD           = 5,           ///< asymmetric motion partition, 2Nx(3N/2) + 2Nx( N/2)
	SIZE_nLx2N           = 6,           ///< asymmetric motion partition, ( N/2)x2N + (3N/2)x2N
	SIZE_nRx2N           = 7,           ///< asymmetric motion partition, (3N/2)x2N + ( N/2)x2N
	NUMBER_OF_PART_SIZES = 8
  };*/
  // uiDepth CU�����
#if AMP_ENC_SPEEDUP
Void TEncCu::xCompressCU(TComDataCU*& rpcBestCU, TComDataCU*& rpcTempCU, const UInt uiDepth DEBUG_STRING_FN_DECLARE(sDebug_), PartSize eParentPartSize)
#else
Void TEncCu::xCompressCU(TComDataCU*& rpcBestCU, TComDataCU*& rpcTempCU, const UInt uiDepth)
#endif
{
	doRMD = false;
	pre0 = -1;		// -1代表不确定正类还是负类，1代表认为是正类，0代表认为是负类
	pre1_1 = -1;
	pre2 = -1;
	pre1_2 = -1;
	TComPic* pcPic = rpcBestCU->getPic();
	DEBUG_STRING_NEW(sDebug)
		const TComPPS& pps = *(rpcTempCU->getSlice()->getPPS());
	const TComSPS& sps = *(rpcTempCU->getSlice()->getSPS());

	// These are only used if getFastDeltaQp() is true
	const UInt fastDeltaQPCuMaxSize = Clip3(sps.getMaxCUHeight() >> sps.getLog2DiffMaxMinCodingBlockSize(), sps.getMaxCUHeight(), 32u);

	m_ppcOrigYuv[uiDepth]->copyFromPicYuv(pcPic->getPicYuvOrg(), rpcBestCU->getCtuRsAddr(), rpcBestCU->getZorderIdxInCtu());

	split = false;
	cease = false;
	for (int depth = 0; depth <= 2; depth++)
	{
		if (uiDepth == depth && isComplete)
		{
			if (depth == 0 && superSplit0)
			{
				pre0 = 1;
				break;
			}
			else if (depth == 2 && superCease2)
			{
				pre2 = 0;
				break;
			}
			int i, j;
			float s = 0;
			UInt uiPartSize = rpcBestCU->getWidth(0);
			const ComponentID compID = ComponentID(0);
			const Int Width = uiPartSize >> m_ppcOrigYuv[uiDepth]->getComponentScaleX(compID);   // 整个CTU的边长
			const Pel* pSrc0 = m_ppcOrigYuv[uiDepth]->getAddr(compID, 0, Width);
			const Int  iSrc0Stride = m_ppcOrigYuv[uiDepth]->getStride(compID);
			vector<vector<int>> p(Width, vector<int>(Width));
			int cntIs0m = 0;
			int cntIs0n = 0;
			// 得到灰度图
			for (i = 0; i < Width; i++)
			{
				for (j = 0; j < Width; j++)
				{
					p[i][j] = pSrc0[j];
					s += pSrc0[j];
					if (j == Width - 1 && p[i][j] == 0)
						cntIs0m++;
					if (i == Width - 1 && p[i][j] == 0)
						cntIs0n++;
				}
				pSrc0 += iSrc0Stride;
				if (cntIs0m >= Width || cntIs0n >= Width)
				{
					isComplete = false;
					break;
				}
			}
			if (!isComplete)
				break;
			if (uiDepth == 2 && !data0)
				doRMD = true;
			ND = calD(p);			// 得到局部方差和平均方差
			SAG = SAGD(p);          // 得到方向复杂度和方向复杂度方差（子CU方向复杂度之间的方差）
			if (!judged_qp)
			{
				qp = m_ppcBestCU[0]->getQP(0);
				judged_qp = true;
			}
			tpSample.assign({ log10(qp), ND[0], ND[1], SAG[0], SAG[1] });
			if (!data0)	// 如果还没有生成模型
			{
				if (depth == 0)
					sample0 = tpSample;
				else if (depth == 1)
					sample1T.push_back(tpSample);
				else if (depth == 2)
					sample2.push_back(tpSample);
			}
			else if (flag[depth] > 0)
			{
				if (tpSample[2] < -10)
					cease = true;
				else if (depth == 0 && !superSplit0 && flag[depth] == 1 && (pre0 = tlinear0.do_predict(tpSample, prob_est)) == 0)
					pre0 = -1;
				else if (depth == 1)
				{
					if (flag[depth] == 1 && (pre1_1 = tlinear1.do_predict(tpSample, prob_est)) == 0)
						pre1_1 = -1;
					else if (flag[depth] == 2 && (pre1_2 = tlinear1.do_predict(tpSample, prob_est)) == 1)
						pre1_2 = -1;
				}
				else if (depth == 2 && flag[depth] == 1 && !superCease2 && (pre2 = tlinear2_90.do_predict(tpSample, prob_est)) == 1)
				{
					if ((pre2 = tlinear2_60.do_predict(tpSample, prob_est)) == 0)
						doRMD = true;
					pre2 = -1;
				}
				else if (depth == 2 && flag[depth] == 2 && isComplete && (pre2 = tlinear2_60.do_predict(tpSample, prob_est)) == 0)
				{
					doRMD = true;
					pre2 = -1;
				}
			}
		}
	}

	split = split || (pre0 == 1 || pre1_1 == 1 || pre1_2 == 1 || pre2 == 1);
	cease = cease || (pre0 == 0 || pre1_1 == 0 || pre1_2 == 0 || pre2 == 0);

	/**********************************************************************/

	// variable for Cbf fast mode PU decision
	Bool    doNotBlockPu = true;
	Bool    earlyDetectionSkipMode = false;

	const UInt uiLPelX = rpcBestCU->getCUPelX();
	const UInt uiRPelX = uiLPelX + rpcBestCU->getWidth(0) - 1;
	const UInt uiTPelY = rpcBestCU->getCUPelY();
	const UInt uiBPelY = uiTPelY + rpcBestCU->getHeight(0) - 1;
	const UInt uiWidth = rpcBestCU->getWidth(0);
	Int iBaseQP = xComputeQP(rpcBestCU, uiDepth);   // ͨ��iBaseQP����iMinQP��iMaxQP
	Int iMinQP;
	Int iMaxQP;
	Bool isAddLowestQP = false;                       // ģʽ����TransquantBypassģʽ

	const UInt numberValidComponents = rpcBestCU->getPic()->getNumberValidComponents();

	if (uiDepth <= pps.getMaxCuDQPDepth())
	{
		Int idQP = m_pcEncCfg->getMaxDeltaQP();
		iMinQP = Clip3(-sps.getQpBDOffset(CHANNEL_TYPE_LUMA), MAX_QP, iBaseQP - idQP);
		iMaxQP = Clip3(-sps.getQpBDOffset(CHANNEL_TYPE_LUMA), MAX_QP, iBaseQP + idQP);
	}
	else
	{
		iMinQP = rpcTempCU->getQP(0);
		iMaxQP = rpcTempCU->getQP(0);
	}
	if (m_pcEncCfg->getUseRateCtrl())
	{
		iMinQP = m_pcRateCtrl->getRCQP();
		iMaxQP = m_pcRateCtrl->getRCQP();
	}

	// transquant-bypass (TQB) processing loop variable initialisation ---

	const Int lowestQP = iMinQP; // For TQB, use this QP which is the lowest non TQB QP tested (rather than QP'=0) - that way delta QPs are smaller, and TQB can be tested at all CU levels.
	if ((pps.getTransquantBypassEnableFlag()))
	{
		isAddLowestQP = true; // mark that the first iteration is to cost TQB mode.�����TQBģʽ�����Ϊtrue
		iMinQP = iMinQP - 1;  // increase loop variable range by 1, to allow testing of TQB mode along with other QPs
		if (m_pcEncCfg->getCUTransquantBypassFlagForceValue())
		{
			iMaxQP = iMinQP;
		}
	}
	// ��ȡ��ǰ����slice
	TComSlice* pcSlice = rpcTempCU->getPic()->getSlice(rpcTempCU->getPic()->getCurrSliceIdx());
	// ��ǰCU����ұ߽�������ͼ������ұ� ���� �±߽�������ͼ�����±� ��ΪTRUE�����ڱ߽磩
	const Bool bBoundary = !(uiRPelX < sps.getPicWidthInLumaSamples() && uiBPelY < sps.getPicHeightInLumaSamples());

	if (!bBoundary && !split)
	{
		for (Int iQP = iMinQP; iQP <= iMaxQP; iQP++)
		{
			const Bool bIsLosslessMode = isAddLowestQP && (iQP == iMinQP);
			if (bIsLosslessMode)
			{
				iQP = lowestQP;
			}
			m_cuChromaQpOffsetIdxPlus1 = 0;
			if (pcSlice->getUseChromaQpAdj())
			{
				/* Pre-estimation of chroma QP based on input block activity may be performed
				 * here, using for example m_ppcOrigYuv[uiDepth] */
				 /* To exercise the current code, the index used for adjustment is based on
				  * block position
				  */
				Int lgMinCuSize = sps.getLog2MinCodingBlockSize() +
					std::max<Int>(0, sps.getLog2DiffMaxMinCodingBlockSize() - Int(pps.getPpsRangeExtension().getDiffCuChromaQpOffsetDepth()));
				m_cuChromaQpOffsetIdxPlus1 = ((uiLPelX >> lgMinCuSize) + (uiTPelY >> lgMinCuSize)) % (pps.getPpsRangeExtension().getChromaQpOffsetListLen() + 1);
			}
			rpcTempCU->initEstData(uiDepth, iQP, bIsLosslessMode);

			// do inter modes, SKIP and 2Nx2N
			// 1������mergeģʽ����xCheckRDCostMerge2Nx2N��2��������ͨ��֡��Ԥ�⣨��AMVP������xCheckRDCostInter
			if (rpcBestCU->getSlice()->getSliceType() != I_SLICE)
			{
				// 2Nx2N
				if (m_pcEncCfg->getUseEarlySkipDetection())  // ʹ��early skip��������ģʽ
				{
					xCheckRDCostInter(rpcBestCU, rpcTempCU, SIZE_2Nx2N DEBUG_STRING_PASS_INTO(sDebug)); // ��������ͨģʽ����Ԥ��
					rpcTempCU->initEstData(uiDepth, iQP, bIsLosslessMode);//by Competition for inter_2Nx2N rpcBestCU�����������ŵ�Ԥ�ⷽʽ�µĲ�����rpcTempCU��ÿ�����ڳ��Ի���Ԥ���CU��ÿ����������½��г�ʼ��
				}
				// SKIP ������Mergeģʽ����Ԥ�⣬��������������ʶ�����ģʽΪskip���޸ĸò���ֵ
				xCheckRDCostMerge2Nx2N(rpcBestCU, rpcTempCU DEBUG_STRING_PASS_INTO(sDebug), &earlyDetectionSkipMode);//by Merge for inter_2Nx2N
				rpcTempCU->initEstData(uiDepth, iQP, bIsLosslessMode);

				if (!m_pcEncCfg->getUseEarlySkipDetection()) //���û��skip
				{
					// 2Nx2N, NxN ��Ȼû�ҵ�NxN���Ķ�
					xCheckRDCostInter(rpcBestCU, rpcTempCU, SIZE_2Nx2N DEBUG_STRING_PASS_INTO(sDebug));
					rpcTempCU->initEstData(uiDepth, iQP, bIsLosslessMode);
					if (m_pcEncCfg->getUseCbfFastMode())   // ʹ�ÿ���cbfģʽ
					{ // Bool doNotBlockPu = true;
						doNotBlockPu = rpcBestCU->getQtRootCbf(0) != 0;   // �ж��Ĳ������ڵ��CBFlag���Ϊtrue������Ҫ������������
					}
				}
			}

			if (bIsLosslessMode) // Restore loop variable if lossless mode was searched.
			{
				iQP = iMinQP;
			}
		}

		if (!earlyDetectionSkipMode) // ���֮ǰû��������ǰ�����������������еĻ��ַ�ʽ��֡��
		{
			for (Int iQP = iMinQP; iQP <= iMaxQP; iQP++)
			{
				// Bool isAddLowestQP = false; Ĭ����false ��455��
				const Bool bIsLosslessMode = isAddLowestQP && (iQP == iMinQP); // If lossless, then iQP is irrelevant for subsequent modules.
				// �����TransquantBypassģʽ��������bIsLosslessMode�����ͱ�ʶ���������ǰö�ٵ���СQP�������ΪlowestQP
				if (bIsLosslessMode)
				{
					iQP = lowestQP;
				}

				rpcTempCU->initEstData(uiDepth, iQP, bIsLosslessMode);

				// do inter modes, NxN, 2NxN, and Nx2N 
				if (rpcBestCU->getSlice()->getSliceType() != I_SLICE)
				{
					// 2Nx2N, NxN

					if (!((rpcBestCU->getWidth(0) == 8) && (rpcBestCU->getHeight(0) == 8)))  // ��ߴ粻��Ϊ8X8 / ��CU���ֵ���С(8X8)
					{
						if (uiDepth == sps.getLog2DiffMaxMinCodingBlockSize() && doNotBlockPu)  //�����ǰ������Ϊ��ǰ���Ĳ����ײ� �Ҳ�������������cbf���� �����NXN����
						{ // ����֡��NXNģʽ���۲��Ƚ�
							xCheckRDCostInter(rpcBestCU, rpcTempCU, SIZE_NxN DEBUG_STRING_PASS_INTO(sDebug));     // ��NXN����ͨԤ��
							rpcTempCU->initEstData(uiDepth, iQP, bIsLosslessMode);
						}
					}
					// NX2N
					if (doNotBlockPu)  // ���������cbf����
					{
						xCheckRDCostInter(rpcBestCU, rpcTempCU, SIZE_Nx2N DEBUG_STRING_PASS_INTO(sDebug));
						rpcTempCU->initEstData(uiDepth, iQP, bIsLosslessMode);
						if (m_pcEncCfg->getUseCbfFastMode() && rpcBestCU->getPartitionSize(0) == SIZE_Nx2N) // ���ʹ�ÿ���CBF���� �ң��ոճ��Եģ�NX2N����ѵĻ���
						{
							doNotBlockPu = rpcBestCU->getQtRootCbf(0) != 0; // �ж��Ĳ������ڵ��CBFlag���Ϊtrue������Ҫ������������
						}
					}
					if (doNotBlockPu)
					{
						xCheckRDCostInter(rpcBestCU, rpcTempCU, SIZE_2NxN DEBUG_STRING_PASS_INTO(sDebug));
						rpcTempCU->initEstData(uiDepth, iQP, bIsLosslessMode);
						if (m_pcEncCfg->getUseCbfFastMode() && rpcBestCU->getPartitionSize(0) == SIZE_2NxN)
						{
							doNotBlockPu = rpcBestCU->getQtRootCbf(0) != 0;
						}
					}
					// 1�����Ȳ���AMP_ENC_SPEEDUP�꣨��ʾ�Ƿ�ӿ�����ٶȣ��Ƿ����2�����AMP_ENC_SPEEDUP�������1��Ĭ������£����TestAMP_Hor��TestAMP_VerΪ�棬��ô���Դ���2NxnU��2NxnD��nLx2N��nRx2N������ģʽ��
					//��2�����TestAMP_Hor��TestAMP_VerΪ�٣����ǿ�����AMP_MRG�꣬����TestMergeAMP_Hor��TestMergeAMP_VerΪ�棬��ô���ǿ��Դ���2NxnU��2NxnD��nLx2N��nRx2N������ģʽ�������ٴ���������ģʽ��
					//��3��������������һЩ�����������Ƿ���Ҫ����2NxnU��2NxnD��nLx2N��nRx2N������ģʽ�����ĳЩʱ���ٶȻ��һ�㡣
					// 3�����AMP_ENC_SPEEDUP�رգ���ôֱ�Ӵ���2NxnU��2NxnD��nLx2N��nRx2N������ģʽ����Ϊû�����������ƣ�������ģʽ��Ҫ���ԣ���ˣ��ٶȻ���һ��
							  //! Try AMP (SIZE_2NxnU, SIZE_2NxnD, SIZE_nLx2N, SIZE_nRx2N)  ���ԷǶԳƷָ�
					if (sps.getUseAMP() && uiDepth < sps.getLog2DiffMaxMinCodingBlockSize())  // �������ǶԳƷָ� �� ��ǰ�����Ȳ��ǵ�ǰ�ĵ��Ĳ����ײ�
					{
#if AMP_ENC_SPEEDUP     // ���AMP������٣������֮ǰ���Ի��ֵ�������ʡȥ����һЩAMP�Ļ���������Դ˴ﵽ�ӿ�����Ŀ��
						Bool bTestAMP_Hor = false, bTestAMP_Ver = false;    // �Ƿ�ʹ��AMP���򻮷֡����򻮷ֵı�ʶ

#if AMP_MRG     // AMP Merge
						Bool bTestMergeAMP_Hor = false, bTestMergeAMP_Ver = false;
						// ����TestAMP_Hor��TestAMP_Ver�Ƿ�Ϊ��
						deriveTestModeAMP(rpcBestCU, eParentPartSize, bTestAMP_Hor, bTestAMP_Ver, bTestMergeAMP_Hor, bTestMergeAMP_Ver);
#else
						deriveTestModeAMP(rpcBestCU, eParentPartSize, bTestAMP_Hor, bTestAMP_Ver);
#endif

						//! Do horizontal AMP
						if (bTestAMP_Hor)     // ���TestAMP_HorΪ�棬����Խ���AMP���򻮷֣���2NxnU��2NxnD�����ֻ���ģʽ
						{
							if (doNotBlockPu)      // ��֮ǰ�ĶԳƻ��ֵ���ͨģʽһ�����������PartSize��Ϊ��Ӧ�ķǶԳƻ���
							{ // ����2NxnUģʽ
								xCheckRDCostInter(rpcBestCU, rpcTempCU, SIZE_2NxnU DEBUG_STRING_PASS_INTO(sDebug));   // �ָ���Ϊ���ߣ�CU��СΪ2N����ָ��߾��붥��N/2������ײ�3N/2
								rpcTempCU->initEstData(uiDepth, iQP, bIsLosslessMode);
								if (m_pcEncCfg->getUseCbfFastMode() && rpcBestCU->getPartitionSize(0) == SIZE_2NxnU)
								{
									doNotBlockPu = rpcBestCU->getQtRootCbf(0) != 0;
								}
							}
							if (doNotBlockPu)
							{ // ����2NxnDģʽ
								xCheckRDCostInter(rpcBestCU, rpcTempCU, SIZE_2NxnD DEBUG_STRING_PASS_INTO(sDebug));   // �ָ���Ϊ���ߣ�����ײ�����
								rpcTempCU->initEstData(uiDepth, iQP, bIsLosslessMode);
								if (m_pcEncCfg->getUseCbfFastMode() && rpcBestCU->getPartitionSize(0) == SIZE_2NxnD)
								{
									doNotBlockPu = rpcBestCU->getQtRootCbf(0) != 0;
								}
							}
						}
#if AMP_MRG
						else if (bTestMergeAMP_Hor)   // TestMergeAMP_HorΪ��Ļ�����ʹ��2NxnU��2NxnD�����ֻ���ģʽ
						{
							if (doNotBlockPu)
							{ // ����2NxnUģʽ
								xCheckRDCostInter(rpcBestCU, rpcTempCU, SIZE_2NxnU DEBUG_STRING_PASS_INTO(sDebug), true);
								rpcTempCU->initEstData(uiDepth, iQP, bIsLosslessMode);
								if (m_pcEncCfg->getUseCbfFastMode() && rpcBestCU->getPartitionSize(0) == SIZE_2NxnU)
								{
									doNotBlockPu = rpcBestCU->getQtRootCbf(0) != 0;
								}
							}
							if (doNotBlockPu)
							{ // ����2NxnDģʽ
								xCheckRDCostInter(rpcBestCU, rpcTempCU, SIZE_2NxnD DEBUG_STRING_PASS_INTO(sDebug), true);
								rpcTempCU->initEstData(uiDepth, iQP, bIsLosslessMode);
								if (m_pcEncCfg->getUseCbfFastMode() && rpcBestCU->getPartitionSize(0) == SIZE_2NxnD)
								{
									doNotBlockPu = rpcBestCU->getQtRootCbf(0) != 0;
								}
							}
						}
#endif

						//! Do horizontal AMP
						if (bTestAMP_Ver) // TestAMP_VerΪ��Ļ�����ʹ��nLx2N��nRx2N�����ֻ���ģʽ
						{
							if (doNotBlockPu)
							{
								xCheckRDCostInter(rpcBestCU, rpcTempCU, SIZE_nLx2N DEBUG_STRING_PASS_INTO(sDebug));
								rpcTempCU->initEstData(uiDepth, iQP, bIsLosslessMode);
								if (m_pcEncCfg->getUseCbfFastMode() && rpcBestCU->getPartitionSize(0) == SIZE_nLx2N)
								{
									doNotBlockPu = rpcBestCU->getQtRootCbf(0) != 0;
								}
							}
							if (doNotBlockPu)
							{
								xCheckRDCostInter(rpcBestCU, rpcTempCU, SIZE_nRx2N DEBUG_STRING_PASS_INTO(sDebug));
								rpcTempCU->initEstData(uiDepth, iQP, bIsLosslessMode);
							}
						}
#if AMP_MRG
						else if (bTestMergeAMP_Ver)
						{
							if (doNotBlockPu)
							{
								xCheckRDCostInter(rpcBestCU, rpcTempCU, SIZE_nLx2N DEBUG_STRING_PASS_INTO(sDebug), true);
								rpcTempCU->initEstData(uiDepth, iQP, bIsLosslessMode);
								if (m_pcEncCfg->getUseCbfFastMode() && rpcBestCU->getPartitionSize(0) == SIZE_nLx2N)
								{
									doNotBlockPu = rpcBestCU->getQtRootCbf(0) != 0;
								}
							}
							if (doNotBlockPu)
							{
								xCheckRDCostInter(rpcBestCU, rpcTempCU, SIZE_nRx2N DEBUG_STRING_PASS_INTO(sDebug), true);
								rpcTempCU->initEstData(uiDepth, iQP, bIsLosslessMode);
							}
						}
#endif

#else
						xCheckRDCostInter(rpcBestCU, rpcTempCU, SIZE_2NxnU);
						rpcTempCU->initEstData(uiDepth, iQP, bIsLosslessMode);
						xCheckRDCostInter(rpcBestCU, rpcTempCU, SIZE_2NxnD);
						rpcTempCU->initEstData(uiDepth, iQP, bIsLosslessMode);
						xCheckRDCostInter(rpcBestCU, rpcTempCU, SIZE_nLx2N);
						rpcTempCU->initEstData(uiDepth, iQP, bIsLosslessMode);

						xCheckRDCostInter(rpcBestCU, rpcTempCU, SIZE_nRx2N);
						rpcTempCU->initEstData(uiDepth, iQP, bIsLosslessMode);

#endif
					} // {������AMP}
				}   // {������֡��Ԥ��}

				// do normal intra modes
				// speedup for inter frames
				Double intraCost = 0.0;
				// {֡��Ԥ��}
				if ((rpcBestCU->getSlice()->getSliceType() == I_SLICE) ||
					((!m_pcEncCfg->getDisableIntraPUsInInterSlices()) && (
						(rpcBestCU->getCbf(0, COMPONENT_Y) != 0) ||
						((rpcBestCU->getCbf(0, COMPONENT_Cb) != 0) && (numberValidComponents > COMPONENT_Cb)) ||
						((rpcBestCU->getCbf(0, COMPONENT_Cr) != 0) && (numberValidComponents > COMPONENT_Cr))  // avoid very complex intra if it is unlikely
						)))     // �����I֡ ���� ������֡��Ԥ����CU�ѱ����CBF��Ԥ��в�Ϊ0��
				{
					xCheckRDCostIntra(rpcBestCU, rpcTempCU, intraCost, SIZE_2Nx2N DEBUG_STRING_PASS_INTO(sDebug));  // ����2NX2N֡��Ԥ�⣬
					rpcTempCU->initEstData(uiDepth, iQP, bIsLosslessMode);
					if (uiDepth == sps.getLog2DiffMaxMinCodingBlockSize())   // �����ǰ���Ϊ�Ĳ�����ײ�
					{
						if (rpcTempCU->getWidth(0) > (1 << sps.getQuadtreeTULog2MinSize()))  // �����ǰCU��ȴ�����СTU���
						{
							if (data0 && doRMD)
								tout << "进来了" << endl;
							Double tmpIntraCost;
							xCheckRDCostIntra(rpcBestCU, rpcTempCU, tmpIntraCost, SIZE_NxN DEBUG_STRING_PASS_INTO(sDebug));   // ����NXN֡��Ԥ��
							intraCost = std::min(intraCost, tmpIntraCost);
							rpcTempCU->initEstData(uiDepth, iQP, bIsLosslessMode);
						}
					}
				}
				if (sps.getUsePCM()
					&& rpcTempCU->getWidth(0) <= (1 << sps.getPCMLog2MaxSize())
					&& rpcTempCU->getWidth(0) >= (1 << sps.getPCMLog2MinSize()))   // �������PCM�ҵ�ǰCU�Ŀ����PCM��С�����Χ��
				{
					UInt uiRawBits = getTotalBits(rpcBestCU->getWidth(0), rpcBestCU->getHeight(0), rpcBestCU->getPic()->getChromaFormat(), sps.getBitDepths().recon); // ֱ�Ӵ�������CU���ص�����
					UInt uiBestBits = rpcBestCU->getTotalBits();  // ��CU�������Ԥ����������
					if ((uiBestBits > uiRawBits) || (rpcBestCU->getTotalCost() > m_pcRdCost->calcRdCost(uiRawBits, 0)))
					{ // �������Ԥ���������ʴ�������CU���ص����� ����ǰ��RDO���ں��ߵ�RDO
						xCheckIntraPCM(rpcBestCU, rpcTempCU);  // ����ʹ��PCMģʽ
						rpcTempCU->initEstData(uiDepth, iQP, bIsLosslessMode);
					} // {������ö��iQP}
				}   // {�������������л��ַ�ʽ}

				if (bIsLosslessMode) // Restore loop variable if lossless mode was searched.
				{
					iQP = iMinQP;
				}
			}
		}

		if (rpcBestCU->getTotalCost() != MAX_DOUBLE) // ���ڲ��Ե�����û�г�����������������ر���
		{
			m_pcRDGoOnSbacCoder->load(m_pppcRDSbacCoder[uiDepth][CI_NEXT_BEST]);
			m_pcEntropyCoder->resetBits();    // ���ñ�����/������
			m_pcEntropyCoder->encodeSplitFlag(rpcBestCU, 0, uiDepth, true);   // �Էָ��־���б���
			rpcBestCU->getTotalBits() += m_pcEntropyCoder->getNumberOfWrittenBits(); // split bits    ��������ͳ��
			rpcBestCU->getTotalBins() += ((TEncBinCABAC*)((TEncSbac*)m_pcEntropyCoder->m_pcEntropyCoderIf)->getEncBinIf())->getBinsCoded();
			rpcBestCU->getTotalCost() = m_pcRdCost->calcRdCost(rpcBestCU->getTotalBits(), rpcBestCU->getTotalDistortion());    // �ܵ�����ͳ��
			m_pcRDGoOnSbacCoder->store(m_pppcRDSbacCoder[uiDepth][CI_NEXT_BEST]); // �����ܵ�RDCost
		}
		//bool res = hytest(rpcBestCU, m_ppcOrigYuv[uiDepth], m_ppcPredYuvBest[uiDepth], 1, rpcBestCU->getTotalCost());
	}

	// copy original YUV samples to PCM buffer
	if (rpcBestCU->getTotalCost() != MAX_DOUBLE && rpcBestCU->isLosslessCoded(0) && (rpcBestCU->getIPCMFlag(0) == false))
	{
		xFillPCMBuffer(rpcBestCU, m_ppcOrigYuv[uiDepth]);   // ��ԭʼYUV�������Ƶ�PCM������
	}

	if (uiDepth == pps.getMaxCuDQPDepth())
	{
		Int idQP = m_pcEncCfg->getMaxDeltaQP();
		iMinQP = Clip3(-sps.getQpBDOffset(CHANNEL_TYPE_LUMA), MAX_QP, iBaseQP - idQP);
		iMaxQP = Clip3(-sps.getQpBDOffset(CHANNEL_TYPE_LUMA), MAX_QP, iBaseQP + idQP);
	}
	else if (uiDepth < pps.getMaxCuDQPDepth())
	{
		iMinQP = iBaseQP;
		iMaxQP = iBaseQP;
	}
	else
	{
		const Int iStartQP = rpcTempCU->getQP(0);
		iMinQP = iStartQP;
		iMaxQP = iStartQP;
	}

	if (m_pcEncCfg->getUseRateCtrl())
	{
		iMinQP = m_pcRateCtrl->getRCQP();
		iMaxQP = m_pcRateCtrl->getRCQP();
	}

	if (m_pcEncCfg->getCUTransquantBypassFlagForceValue())
	{
		iMaxQP = iMinQP; // If all TUs are forced into using transquant bypass, do not loop here.
	}

	const Bool bSubBranch = bBoundary || !(m_pcEncCfg->getUseEarlyCU() && rpcBestCU->getTotalCost() != MAX_DOUBLE && rpcBestCU->isSkipped(0));

	if (!cease && bSubBranch && uiDepth < sps.getLog2DiffMaxMinCodingBlockSize() && (!getFastDeltaQp() || uiWidth > fastDeltaQPCuMaxSize || bBoundary))
	{
		// further split ����С������������������������ݹ鴦����CU��Ȼ��ѡȡ���ŵ��������������Ż���ģʽ
		for (Int iQP = iMinQP; iQP <= iMaxQP; iQP++)
		{
			const Bool bIsLosslessMode = false; // False at this level. Next level down may set it to true.

			rpcTempCU->initEstData(uiDepth, iQP, bIsLosslessMode);

			UChar       uhNextDepth = uiDepth + 1;                  // ��һ������
			TComDataCU* pcSubBestPartCU = m_ppcBestCU[uhNextDepth];   // ��һ������CU����
			TComDataCU* pcSubTempPartCU = m_ppcTempCU[uhNextDepth];   // ��һ�����ʱCU����
			DEBUG_STRING_NEW(sTempDebug)
				// ��һ���ָ��ǰCU������Ϊ4����CU
				for (UInt uiPartUnitIdx = 0; uiPartUnitIdx < 4; uiPartUnitIdx++)    // ö�ٻ����Ĳ������ĸ��ӿ���±�
				{
					pcSubBestPartCU->initSubCU(rpcTempCU, uiPartUnitIdx, uhNextDepth, iQP);           // clear sub partition datas or init.��ջ��ʼ��BestCU�ӿ������
					pcSubTempPartCU->initSubCU(rpcTempCU, uiPartUnitIdx, uhNextDepth, iQP);           // clear sub partition datas or init.��ջ��ʼ��TempCU�ӿ������
					// ����ӿ�CU�ĺ�������λ������������ͼ��֮�ڣ����Լ������µ�����
					if ((pcSubBestPartCU->getCUPelX() < sps.getPicWidthInLumaSamples()) && (pcSubBestPartCU->getCUPelY() < sps.getPicHeightInLumaSamples()))
					{
						if (0 == uiPartUnitIdx) //initialize RD with previous depth buffer   ����Ĵ�����һ���ӿ飨���Ͻǣ�
						{
							m_pppcRDSbacCoder[uhNextDepth][CI_CURR_BEST]->load(m_pppcRDSbacCoder[uiDepth][CI_CURR_BEST]);   // ʹ��֮ǰ����ǰ��ȣ��Ļ����ʼ��RDO
						}
						else  // ���������ӿ�
						{
							m_pppcRDSbacCoder[uhNextDepth][CI_CURR_BEST]->load(m_pppcRDSbacCoder[uhNextDepth][CI_NEXT_BEST]);   // ʹ�õ�ǰ����һ��ȣ��Ļ����ʼ��RDO
						}
						/*
						  Bool          isInter                       ( UInt uiPartIdx ) const                                     { return m_pePredMode[ uiPartIdx ] == MODE_INTER; }
						enum PredMode
						{
						  MODE_INTER                 = 0,     ///< inter-prediction mode
						  MODE_INTRA                 = 1,     ///< intra-prediction mode
						  NUMBER_OF_PREDICTION_MODES = 2,
						};
						*/
#if AMP_ENC_SPEEDUP
						DEBUG_STRING_NEW(sChild)
							if (!(rpcBestCU->getTotalCost() != MAX_DOUBLE && rpcBestCU->isInter(0)))
							{
								xCompressCU(pcSubBestPartCU, pcSubTempPartCU, uhNextDepth DEBUG_STRING_PASS_INTO(sChild), NUMBER_OF_PART_SIZES);
							}
							else
							{

								xCompressCU(pcSubBestPartCU, pcSubTempPartCU, uhNextDepth DEBUG_STRING_PASS_INTO(sChild), rpcBestCU->getPartitionSize(0));
							}
						DEBUG_STRING_APPEND(sTempDebug, sChild)
#else
						xCompressCU(pcSubBestPartCU, pcSubTempPartCU, uhNextDepth); // �ݹ���һ����ӿ�
#endif

						rpcTempCU->copyPartFrom(pcSubBestPartCU, uiPartUnitIdx, uhNextDepth);           // Keep best part data to current temporary data.����õ��ӿ�����ݴ��ڵ�ǰ����ʱ������
						xCopyYuv2Tmp(pcSubBestPartCU->getTotalNumPart() * uiPartUnitIdx, uhNextDepth);    // ����Ԥ��ͼ����ؽ�ͼ���YUV����
					}   // {���������Լ������µ���}
					else
					{
						pcSubBestPartCU->copyToPic(uhNextDepth);    // ����ǰԤ��Ĳ��ָ��Ƶ�ͼƬ�е�CU������Ԥ����һ���ӿ�
						rpcTempCU->copyPartFrom(pcSubBestPartCU, uiPartUnitIdx, uhNextDepth);   // ����õ����ݴ��ڵ�ǰ����ʱ������
					}
				}

			m_pcRDGoOnSbacCoder->load(m_pppcRDSbacCoder[uhNextDepth][CI_NEXT_BEST]);
			if (!bBoundary)  // �����ǰ�鲻�ڱ߽磬�����ر���
			{
				m_pcEntropyCoder->resetBits();  // ��������
				m_pcEntropyCoder->encodeSplitFlag(rpcTempCU, 0, uiDepth, true);   // �Էָ��־���б���

				rpcTempCU->getTotalBits() += m_pcEntropyCoder->getNumberOfWrittenBits(); // split bits
				rpcTempCU->getTotalBins() += ((TEncBinCABAC*)((TEncSbac*)m_pcEntropyCoder->m_pcEntropyCoderIf)->getEncBinIf())->getBinsCoded();    // �����ر�������
			}
			rpcTempCU->getTotalCost() = m_pcRdCost->calcRdCost(rpcTempCU->getTotalBits(), rpcTempCU->getTotalDistortion());    // �����ܵ�RDCost

			if (uiDepth == pps.getMaxCuDQPDepth() && pps.getUseDQP()) // ���ʹ��DeltaQP�ҵ�ǰ��ȵ���DeltaQP������
			{
				Bool hasResidual = false;   // �ܷ��вв�ı�ʶ
				for (UInt uiBlkIdx = 0; uiBlkIdx < rpcTempCU->getTotalNumPart(); uiBlkIdx++)   // ö�����л��ֵ���С��Cuba��
				{
					if ((rpcTempCU->getCbf(uiBlkIdx, COMPONENT_Y)
						|| (rpcTempCU->getCbf(uiBlkIdx, COMPONENT_Cb) && (numberValidComponents > COMPONENT_Cb))
						|| (rpcTempCU->getCbf(uiBlkIdx, COMPONENT_Cr) && (numberValidComponents > COMPONENT_Cr))))
					{ // Cbf!=0�����вв�
						hasResidual = true; // ��ʶ�вв�Ϊtrue
						break;
					}
				}

				if (hasResidual)  // ����вв�����ر���
				{
					m_pcEntropyCoder->resetBits();
					m_pcEntropyCoder->encodeQP(rpcTempCU, 0, false);
					rpcTempCU->getTotalBits() += m_pcEntropyCoder->getNumberOfWrittenBits(); // dQP bits
					rpcTempCU->getTotalBins() += ((TEncBinCABAC*)((TEncSbac*)m_pcEntropyCoder->m_pcEntropyCoderIf)->getEncBinIf())->getBinsCoded();
					rpcTempCU->getTotalCost() = m_pcRdCost->calcRdCost(rpcTempCU->getTotalBits(), rpcTempCU->getTotalDistortion());

					Bool foundNonZeroCbf = false; // �ҵ�����cbf��ʶ
					rpcTempCU->setQPSubCUs(rpcTempCU->getRefQP(0), 0, uiDepth, foundNonZeroCbf);  // �����ӿ�QP
					assert(foundNonZeroCbf);
				}
				else  // ������СCU��û�вв�
				{
					rpcTempCU->setQPSubParts(rpcTempCU->getRefQP(0), 0, uiDepth); // set QP to default QP ���ӿ�QP����ΪĬ��ֵ
				}
			} // {����������DeltaQP���}

			m_pcRDGoOnSbacCoder->store(m_pppcRDSbacCoder[uiDepth][CI_TEMP_BEST]); // ���浱ǰ��Ȼ������ʱ����RDCost

			// If the configuration being tested exceeds the maximum number of bytes for a slice / slice-segment, then
			// a proper RD evaluation cannot be performed. Therefore, termination of the
			// slice/slice-segment must be made prior to this CTU.
			// This can be achieved by forcing the decision to be that of the rpcTempCU.
			// The exception is each slice / slice-segment must have at least one CTU.
			if (rpcBestCU->getTotalCost() != MAX_DOUBLE)    // ���ڲ��Ե�����û�г�������ֽ���
			{
				const Bool isEndOfSlice = pcSlice->getSliceMode() == FIXED_NUMBER_OF_BYTES  // �Ƿ���Slice��ĩ�ı�ʶ
					&& ((pcSlice->getSliceBits() + rpcBestCU->getTotalBits()) > pcSlice->getSliceArgument() << 3)
					&& rpcBestCU->getCtuRsAddr() != pcPic->getPicSym()->getCtuTsToRsAddrMap(pcSlice->getSliceCurStartCtuTsAddr())
					&& rpcBestCU->getCtuRsAddr() != pcPic->getPicSym()->getCtuTsToRsAddrMap(pcSlice->getSliceSegmentCurStartCtuTsAddr());
				const Bool isEndOfSliceSegment = pcSlice->getSliceSegmentMode() == FIXED_NUMBER_OF_BYTES   // �Ƿ���SS��ĩ�ı�ʶ
					&& ((pcSlice->getSliceSegmentBits() + rpcBestCU->getTotalBits()) > pcSlice->getSliceSegmentArgument() << 3)
					&& rpcBestCU->getCtuRsAddr() != pcPic->getPicSym()->getCtuTsToRsAddrMap(pcSlice->getSliceSegmentCurStartCtuTsAddr());
				// Do not need to check slice condition for slice-segment since a slice-segment is a subset of a slice.
				if (isEndOfSlice || isEndOfSliceSegment)   // ������Ƭ������Ƭ���Ӽ�����˲���Ҫ�����Ƭ�ε���Ƭ����
				{
					rpcBestCU->getTotalCost() = MAX_DOUBLE; // �������ĩ�ˣ���RDCost����Ϊ����ֽ���
				}
			}
			sflag = false;
			// 更新最佳模式
			xCheckBestMode(rpcBestCU, rpcTempCU, uiDepth DEBUG_STRING_PASS_INTO(sDebug) DEBUG_STRING_PASS_INTO(sTempDebug) DEBUG_STRING_PASS_INTO(false)); // RD compare current larger prediction
																																							 // with sub partitioned prediction.
			if (uiDepth == 0)
				sInfo0 = sflag;
			else if (uiDepth == 1)
				sInfo1.push_back(sflag);
			else if (uiDepth == 2)
				sInfo2.push_back(sflag);
		}
	}

	DEBUG_STRING_APPEND(sDebug_, sDebug);

	rpcBestCU->copyToPic(uiDepth);    // Copy Best data to Picture for next partition prediction.������÷�ʽ������������һ�����Ԥ��
	//if (uiDepth == 1)
	//	indexD1++;
	//else if (uiDepth == 2)
	//	indexD2++;
	//else if (uiDepth == 3)
	//	indexD3++;
	xCopyYuv2Pic(rpcBestCU->getPic(), rpcBestCU->getCtuRsAddr(), rpcBestCU->getZorderIdxInCtu(), uiDepth, uiDepth);   // Copy Yuv data to picture Yuv
	if (bBoundary)
	{
		return;
	}

	// Assert if Best prediction mode is NONE
	// Selected mode's RD-cost must be not MAX_DOUBLE.
	assert(rpcBestCU->getPartitionSize(0) != NUMBER_OF_PART_SIZES);
	assert(rpcBestCU->getPredictionMode(0) != NUMBER_OF_PREDICTION_MODES);
	assert(rpcBestCU->getTotalCost() != MAX_DOUBLE);
}

/** finish encoding a cu and handle end-of-slice conditions
 * \param pcCU
 * \param uiAbsPartIdx
 * \param uiDepth
 * \returns Void
 */
Void TEncCu::finishCU(TComDataCU* pcCU, UInt uiAbsPartIdx)
{
	TComPic* pcPic = pcCU->getPic();
	TComSlice* pcSlice = pcCU->getPic()->getSlice(pcCU->getPic()->getCurrSliceIdx());

	//Calculate end address
	const Int  currentCTUTsAddr = pcPic->getPicSym()->getCtuRsToTsAddrMap(pcCU->getCtuRsAddr());
	const Bool isLastSubCUOfCtu = pcCU->isLastSubCUOfCtu(uiAbsPartIdx);
	if (isLastSubCUOfCtu)
	{
		// The 1-terminating bit is added to all streams, so don't add it here when it's 1.
		// i.e. when the slice segment CurEnd CTU address is the current CTU address+1.
		if (pcSlice->getSliceSegmentCurEndCtuTsAddr() != currentCTUTsAddr + 1)
		{
			m_pcEntropyCoder->encodeTerminatingBit(0);
		}
	}
}

/** Compute QP for each CU
 * \param pcCU Target CU
 * \param uiDepth CU depth
 * \returns quantization parameter
 */
Int TEncCu::xComputeQP(TComDataCU* pcCU, UInt uiDepth)
{
	Int iBaseQp = pcCU->getSlice()->getSliceQp();
	Int iQpOffset = 0;
	if (m_pcEncCfg->getUseAdaptiveQP())
	{
		TEncPic* pcEPic = dynamic_cast<TEncPic*>(pcCU->getPic());
		UInt uiAQDepth = min(uiDepth, pcEPic->getMaxAQDepth() - 1);
		TEncPicQPAdaptationLayer* pcAQLayer = pcEPic->getAQLayer(uiAQDepth);
		UInt uiAQUPosX = pcCU->getCUPelX() / pcAQLayer->getAQPartWidth();
		UInt uiAQUPosY = pcCU->getCUPelY() / pcAQLayer->getAQPartHeight();
		UInt uiAQUStride = pcAQLayer->getAQPartStride();
		TEncQPAdaptationUnit* acAQU = pcAQLayer->getQPAdaptationUnit();

		Double dMaxQScale = pow(2.0, m_pcEncCfg->getQPAdaptationRange() / 6.0);
		Double dAvgAct = pcAQLayer->getAvgActivity();
		Double dCUAct = acAQU[uiAQUPosY * uiAQUStride + uiAQUPosX].getActivity();
		Double dNormAct = (dMaxQScale * dCUAct + dAvgAct) / (dCUAct + dMaxQScale * dAvgAct);
		Double dQpOffset = log(dNormAct) / log(2.0) * 6.0;
		iQpOffset = Int(floor(dQpOffset + 0.49999));
	}

	return Clip3(-pcCU->getSlice()->getSPS()->getQpBDOffset(CHANNEL_TYPE_LUMA), MAX_QP, iBaseQp + iQpOffset);
}

/** encode a CU block recursively
 * \param pcCU
 * \param uiAbsPartIdx
 * \param uiDepth
 * \returns Void
 */
Void TEncCu::xEncodeCU(TComDataCU* pcCU, UInt uiAbsPartIdx, UInt uiDepth)
{
	TComPic* const pcPic = pcCU->getPic();
	TComSlice* const pcSlice = pcCU->getSlice();
	const TComSPS& sps = *(pcSlice->getSPS());
	const TComPPS& pps = *(pcSlice->getPPS());

	const UInt maxCUWidth = sps.getMaxCUWidth();
	const UInt maxCUHeight = sps.getMaxCUHeight();

	Bool bBoundary = false;
	UInt uiLPelX = pcCU->getCUPelX() + g_auiRasterToPelX[g_auiZscanToRaster[uiAbsPartIdx]];
	const UInt uiRPelX = uiLPelX + (maxCUWidth >> uiDepth) - 1;
	UInt uiTPelY = pcCU->getCUPelY() + g_auiRasterToPelY[g_auiZscanToRaster[uiAbsPartIdx]];
	const UInt uiBPelY = uiTPelY + (maxCUHeight >> uiDepth) - 1;

	if ((uiRPelX < sps.getPicWidthInLumaSamples()) && (uiBPelY < sps.getPicHeightInLumaSamples()))
	{
		m_pcEntropyCoder->encodeSplitFlag(pcCU, uiAbsPartIdx, uiDepth);
	}
	else
	{
		bBoundary = true;
	}

	if (((uiDepth < pcCU->getDepth(uiAbsPartIdx)) && (uiDepth < sps.getLog2DiffMaxMinCodingBlockSize())) || bBoundary)
	{
		UInt uiQNumParts = (pcPic->getNumPartitionsInCtu() >> (uiDepth << 1)) >> 2;
		if (uiDepth == pps.getMaxCuDQPDepth() && pps.getUseDQP())
		{
			setdQPFlag(true);
		}

		if (uiDepth == pps.getPpsRangeExtension().getDiffCuChromaQpOffsetDepth() && pcSlice->getUseChromaQpAdj())
		{
			setCodeChromaQpAdjFlag(true);
		}

		for (UInt uiPartUnitIdx = 0; uiPartUnitIdx < 4; uiPartUnitIdx++, uiAbsPartIdx += uiQNumParts)
		{
			uiLPelX = pcCU->getCUPelX() + g_auiRasterToPelX[g_auiZscanToRaster[uiAbsPartIdx]];
			uiTPelY = pcCU->getCUPelY() + g_auiRasterToPelY[g_auiZscanToRaster[uiAbsPartIdx]];
			if ((uiLPelX < sps.getPicWidthInLumaSamples()) && (uiTPelY < sps.getPicHeightInLumaSamples()))
			{
				xEncodeCU(pcCU, uiAbsPartIdx, uiDepth + 1);
			}
		}
		return;
	}

	if (uiDepth <= pps.getMaxCuDQPDepth() && pps.getUseDQP())
	{
		setdQPFlag(true);
	}

	if (uiDepth <= pps.getPpsRangeExtension().getDiffCuChromaQpOffsetDepth() && pcSlice->getUseChromaQpAdj())
	{
		setCodeChromaQpAdjFlag(true);
	}

	if (pps.getTransquantBypassEnableFlag())
	{
		m_pcEntropyCoder->encodeCUTransquantBypassFlag(pcCU, uiAbsPartIdx);
	}

	if (!pcSlice->isIntra())
	{
		m_pcEntropyCoder->encodeSkipFlag(pcCU, uiAbsPartIdx);
	}

	if (pcCU->isSkipped(uiAbsPartIdx))
	{
		m_pcEntropyCoder->encodeMergeIndex(pcCU, uiAbsPartIdx);
		finishCU(pcCU, uiAbsPartIdx);
		return;
	}

	m_pcEntropyCoder->encodePredMode(pcCU, uiAbsPartIdx);
	m_pcEntropyCoder->encodePartSize(pcCU, uiAbsPartIdx, uiDepth);

	if (pcCU->isIntra(uiAbsPartIdx) && pcCU->getPartitionSize(uiAbsPartIdx) == SIZE_2Nx2N)
	{
		m_pcEntropyCoder->encodeIPCMInfo(pcCU, uiAbsPartIdx);

		if (pcCU->getIPCMFlag(uiAbsPartIdx))
		{
			// Encode slice finish
			finishCU(pcCU, uiAbsPartIdx);
			return;
		}
	}

	// prediction Info ( Intra : direction mode, Inter : Mv, reference idx )
	m_pcEntropyCoder->encodePredInfo(pcCU, uiAbsPartIdx);

	// Encode Coefficients
	Bool bCodeDQP = getdQPFlag();
	Bool codeChromaQpAdj = getCodeChromaQpAdjFlag();
	m_pcEntropyCoder->encodeCoeff(pcCU, uiAbsPartIdx, uiDepth, bCodeDQP, codeChromaQpAdj);
	setCodeChromaQpAdjFlag(codeChromaQpAdj);
	setdQPFlag(bCodeDQP);

	// --- write terminating bit ---
	finishCU(pcCU, uiAbsPartIdx);
}

Int xCalcHADs8x8_ISlice(Pel* piOrg, Int iStrideOrg)
{
	Int k, i, j, jj;
	Int diff[64], m1[8][8], m2[8][8], m3[8][8], iSumHad = 0;

	for (k = 0; k < 64; k += 8)
	{
		diff[k + 0] = piOrg[0];
		diff[k + 1] = piOrg[1];
		diff[k + 2] = piOrg[2];
		diff[k + 3] = piOrg[3];
		diff[k + 4] = piOrg[4];
		diff[k + 5] = piOrg[5];
		diff[k + 6] = piOrg[6];
		diff[k + 7] = piOrg[7];

		piOrg += iStrideOrg;
	}

	//horizontal
	for (j = 0; j < 8; j++)
	{
		jj = j << 3;
		m2[j][0] = diff[jj] + diff[jj + 4];
		m2[j][1] = diff[jj + 1] + diff[jj + 5];
		m2[j][2] = diff[jj + 2] + diff[jj + 6];
		m2[j][3] = diff[jj + 3] + diff[jj + 7];
		m2[j][4] = diff[jj] - diff[jj + 4];
		m2[j][5] = diff[jj + 1] - diff[jj + 5];
		m2[j][6] = diff[jj + 2] - diff[jj + 6];
		m2[j][7] = diff[jj + 3] - diff[jj + 7];

		m1[j][0] = m2[j][0] + m2[j][2];
		m1[j][1] = m2[j][1] + m2[j][3];
		m1[j][2] = m2[j][0] - m2[j][2];
		m1[j][3] = m2[j][1] - m2[j][3];
		m1[j][4] = m2[j][4] + m2[j][6];
		m1[j][5] = m2[j][5] + m2[j][7];
		m1[j][6] = m2[j][4] - m2[j][6];
		m1[j][7] = m2[j][5] - m2[j][7];

		m2[j][0] = m1[j][0] + m1[j][1];
		m2[j][1] = m1[j][0] - m1[j][1];
		m2[j][2] = m1[j][2] + m1[j][3];
		m2[j][3] = m1[j][2] - m1[j][3];
		m2[j][4] = m1[j][4] + m1[j][5];
		m2[j][5] = m1[j][4] - m1[j][5];
		m2[j][6] = m1[j][6] + m1[j][7];
		m2[j][7] = m1[j][6] - m1[j][7];
	}

	//vertical
	for (i = 0; i < 8; i++)
	{
		m3[0][i] = m2[0][i] + m2[4][i];
		m3[1][i] = m2[1][i] + m2[5][i];
		m3[2][i] = m2[2][i] + m2[6][i];
		m3[3][i] = m2[3][i] + m2[7][i];
		m3[4][i] = m2[0][i] - m2[4][i];
		m3[5][i] = m2[1][i] - m2[5][i];
		m3[6][i] = m2[2][i] - m2[6][i];
		m3[7][i] = m2[3][i] - m2[7][i];

		m1[0][i] = m3[0][i] + m3[2][i];
		m1[1][i] = m3[1][i] + m3[3][i];
		m1[2][i] = m3[0][i] - m3[2][i];
		m1[3][i] = m3[1][i] - m3[3][i];
		m1[4][i] = m3[4][i] + m3[6][i];
		m1[5][i] = m3[5][i] + m3[7][i];
		m1[6][i] = m3[4][i] - m3[6][i];
		m1[7][i] = m3[5][i] - m3[7][i];

		m2[0][i] = m1[0][i] + m1[1][i];
		m2[1][i] = m1[0][i] - m1[1][i];
		m2[2][i] = m1[2][i] + m1[3][i];
		m2[3][i] = m1[2][i] - m1[3][i];
		m2[4][i] = m1[4][i] + m1[5][i];
		m2[5][i] = m1[4][i] - m1[5][i];
		m2[6][i] = m1[6][i] + m1[7][i];
		m2[7][i] = m1[6][i] - m1[7][i];
	}

	for (i = 0; i < 8; i++)
	{
		for (j = 0; j < 8; j++)
		{
			iSumHad += abs(m2[i][j]);
		}
	}
	iSumHad -= abs(m2[0][0]);
	iSumHad = (iSumHad + 2) >> 2;
	return(iSumHad);
}

Int  TEncCu::updateCtuDataISlice(TComDataCU* pCtu, Int width, Int height)
{
	Int  xBl, yBl;
	const Int iBlkSize = 8;

	Pel* pOrgInit = pCtu->getPic()->getPicYuvOrg()->getAddr(COMPONENT_Y, pCtu->getCtuRsAddr(), 0);
	Int  iStrideOrig = pCtu->getPic()->getPicYuvOrg()->getStride(COMPONENT_Y);
	Pel* pOrg;

	Int iSumHad = 0;
	for (yBl = 0; (yBl + iBlkSize) <= height; yBl += iBlkSize)
	{
		for (xBl = 0; (xBl + iBlkSize) <= width; xBl += iBlkSize)
		{
			pOrg = pOrgInit + iStrideOrig * yBl + xBl;
			iSumHad += xCalcHADs8x8_ISlice(pOrg, iStrideOrig);
		}
	}
	return(iSumHad);
}

/** check RD costs for a CU block encoded with merge
 * \param rpcBestCU
 * \param rpcTempCU
 * \param earlyDetectionSkipMode
 */
Void TEncCu::xCheckRDCostMerge2Nx2N(TComDataCU*& rpcBestCU, TComDataCU*& rpcTempCU DEBUG_STRING_FN_DECLARE(sDebug), Bool* earlyDetectionSkipMode)
{
	assert(rpcTempCU->getSlice()->getSliceType() != I_SLICE);
	if (getFastDeltaQp())
	{
		return;   // never check merge in fast deltaqp mode
	}
	TComMvField  cMvFieldNeighbours[2 * MRG_MAX_NUM_CANDS]; // double length for mv of both lists
	UChar uhInterDirNeighbours[MRG_MAX_NUM_CANDS];
	Int numValidMergeCand = 0;
	const Bool bTransquantBypassFlag = rpcTempCU->getCUTransquantBypass(0);

	for (UInt ui = 0; ui < rpcTempCU->getSlice()->getMaxNumMergeCand(); ++ui)
	{
		uhInterDirNeighbours[ui] = 0;
	}
	UChar uhDepth = rpcTempCU->getDepth(0);
	rpcTempCU->setPartSizeSubParts(SIZE_2Nx2N, 0, uhDepth); // interprets depth relative to CTU level
	rpcTempCU->getInterMergeCandidates(0, 0, cMvFieldNeighbours, uhInterDirNeighbours, numValidMergeCand);

	Int mergeCandBuffer[MRG_MAX_NUM_CANDS];
	for (UInt ui = 0; ui < numValidMergeCand; ++ui)
	{
		mergeCandBuffer[ui] = 0;
	}

	Bool bestIsSkip = false;

	UInt iteration;
	if (rpcTempCU->isLosslessCoded(0))
	{
		iteration = 1;
	}
	else
	{
		iteration = 2;
	}
	DEBUG_STRING_NEW(bestStr)

		for (UInt uiNoResidual = 0; uiNoResidual < iteration; ++uiNoResidual)
		{
			for (UInt uiMergeCand = 0; uiMergeCand < numValidMergeCand; ++uiMergeCand)
			{
				if (!(uiNoResidual == 1 && mergeCandBuffer[uiMergeCand] == 1))
				{
					if (!(bestIsSkip && uiNoResidual == 0))
					{
						DEBUG_STRING_NEW(tmpStr)
							// set MC parameters
							rpcTempCU->setPredModeSubParts(MODE_INTER, 0, uhDepth); // interprets depth relative to CTU level
						rpcTempCU->setCUTransquantBypassSubParts(bTransquantBypassFlag, 0, uhDepth);
						rpcTempCU->setChromaQpAdjSubParts(bTransquantBypassFlag ? 0 : m_cuChromaQpOffsetIdxPlus1, 0, uhDepth);
						rpcTempCU->setPartSizeSubParts(SIZE_2Nx2N, 0, uhDepth); // interprets depth relative to CTU level
						rpcTempCU->setMergeFlagSubParts(true, 0, 0, uhDepth); // interprets depth relative to CTU level
						rpcTempCU->setMergeIndexSubParts(uiMergeCand, 0, 0, uhDepth); // interprets depth relative to CTU level
						rpcTempCU->setInterDirSubParts(uhInterDirNeighbours[uiMergeCand], 0, 0, uhDepth); // interprets depth relative to CTU level
						rpcTempCU->getCUMvField(REF_PIC_LIST_0)->setAllMvField(cMvFieldNeighbours[0 + 2 * uiMergeCand], SIZE_2Nx2N, 0, 0); // interprets depth relative to rpcTempCU level
						rpcTempCU->getCUMvField(REF_PIC_LIST_1)->setAllMvField(cMvFieldNeighbours[1 + 2 * uiMergeCand], SIZE_2Nx2N, 0, 0); // interprets depth relative to rpcTempCU level

						// do MC
						m_pcPredSearch->motionCompensation(rpcTempCU, m_ppcPredYuvTemp[uhDepth]);
						// estimate residual and encode everything
						m_pcPredSearch->encodeResAndCalcRdInterCU(rpcTempCU,
							m_ppcOrigYuv[uhDepth],
							m_ppcPredYuvTemp[uhDepth],
							m_ppcResiYuvTemp[uhDepth],
							m_ppcResiYuvBest[uhDepth],
							m_ppcRecoYuvTemp[uhDepth],
							(uiNoResidual != 0) DEBUG_STRING_PASS_INTO(tmpStr));

#if DEBUG_STRING
						DebugInterPredResiReco(tmpStr, *(m_ppcPredYuvTemp[uhDepth]), *(m_ppcResiYuvBest[uhDepth]), *(m_ppcRecoYuvTemp[uhDepth]), DebugStringGetPredModeMask(rpcTempCU->getPredictionMode(0)));
#endif

						if ((uiNoResidual == 0) && (rpcTempCU->getQtRootCbf(0) == 0))
						{
							// If no residual when allowing for one, then set mark to not try case where residual is forced to 0
							mergeCandBuffer[uiMergeCand] = 1;
						}

						Int orgQP = rpcTempCU->getQP(0);
						xCheckDQP(rpcTempCU);
						xCheckBestMode(rpcBestCU, rpcTempCU, uhDepth DEBUG_STRING_PASS_INTO(bestStr) DEBUG_STRING_PASS_INTO(tmpStr));

						rpcTempCU->initEstData(uhDepth, orgQP, bTransquantBypassFlag);

						if (m_pcEncCfg->getUseFastDecisionForMerge() && !bestIsSkip)
						{
							bestIsSkip = rpcBestCU->getQtRootCbf(0) == 0;
						}
					}
				}
			}

			if (uiNoResidual == 0 && m_pcEncCfg->getUseEarlySkipDetection())
			{
				if (rpcBestCU->getQtRootCbf(0) == 0)
				{
					if (rpcBestCU->getMergeFlag(0))
					{
						*earlyDetectionSkipMode = true;
					}
					else if (m_pcEncCfg->getMotionEstimationSearchMethod() != MESEARCH_SELECTIVE)
					{
						Int absoulte_MV = 0;
						for (UInt uiRefListIdx = 0; uiRefListIdx < 2; uiRefListIdx++)
						{
							if (rpcBestCU->getSlice()->getNumRefIdx(RefPicList(uiRefListIdx)) > 0)
							{
								TComCUMvField* pcCUMvField = rpcBestCU->getCUMvField(RefPicList(uiRefListIdx));
								Int iHor = pcCUMvField->getMvd(0).getAbsHor();
								Int iVer = pcCUMvField->getMvd(0).getAbsVer();
								absoulte_MV += iHor + iVer;
							}
						}

						if (absoulte_MV == 0)
						{
							*earlyDetectionSkipMode = true;
						}
					}
				}
			}
		}
	DEBUG_STRING_APPEND(sDebug, bestStr)
}


#if AMP_MRG
Void TEncCu::xCheckRDCostInter(TComDataCU*& rpcBestCU, TComDataCU*& rpcTempCU, PartSize ePartSize DEBUG_STRING_FN_DECLARE(sDebug), Bool bUseMRG)
#else
Void TEncCu::xCheckRDCostInter(TComDataCU*& rpcBestCU, TComDataCU*& rpcTempCU, PartSize ePartSize)
#endif
{

	DEBUG_STRING_NEW(sTest)
		if (getFastDeltaQp())
		{
			const TComSPS& sps = *(rpcTempCU->getSlice()->getSPS());
			const UInt fastDeltaQPCuMaxSize = Clip3(sps.getMaxCUHeight() >> (sps.getLog2DiffMaxMinCodingBlockSize()), sps.getMaxCUHeight(), 32u);
			if (ePartSize != SIZE_2Nx2N || rpcTempCU->getWidth(0) > fastDeltaQPCuMaxSize)
			{
				return; // only check necessary 2Nx2N Inter in fast deltaqp mode
			}
		}

	// prior to this, rpcTempCU will have just been reset using rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
	UChar uhDepth = rpcTempCU->getDepth(0);

	rpcTempCU->setPartSizeSubParts(ePartSize, 0, uhDepth);
	rpcTempCU->setPredModeSubParts(MODE_INTER, 0, uhDepth);
	rpcTempCU->setChromaQpAdjSubParts(rpcTempCU->getCUTransquantBypass(0) ? 0 : m_cuChromaQpOffsetIdxPlus1, 0, uhDepth);

#if AMP_MRG
	rpcTempCU->setMergeAMP(true);
	m_pcPredSearch->predInterSearch(rpcTempCU, m_ppcOrigYuv[uhDepth], m_ppcPredYuvTemp[uhDepth], m_ppcResiYuvTemp[uhDepth], m_ppcRecoYuvTemp[uhDepth] DEBUG_STRING_PASS_INTO(sTest), false, bUseMRG);
#else
	m_pcPredSearch->predInterSearch(rpcTempCU, m_ppcOrigYuv[uhDepth], m_ppcPredYuvTemp[uhDepth], m_ppcResiYuvTemp[uhDepth], m_ppcRecoYuvTemp[uhDepth]);
#endif

#if AMP_MRG
	if (!rpcTempCU->getMergeAMP())
	{
		return;
	}
#endif

	m_pcPredSearch->encodeResAndCalcRdInterCU(rpcTempCU, m_ppcOrigYuv[uhDepth], m_ppcPredYuvTemp[uhDepth], m_ppcResiYuvTemp[uhDepth], m_ppcResiYuvBest[uhDepth], m_ppcRecoYuvTemp[uhDepth], false DEBUG_STRING_PASS_INTO(sTest));
	rpcTempCU->getTotalCost() = m_pcRdCost->calcRdCost(rpcTempCU->getTotalBits(), rpcTempCU->getTotalDistortion());

#if DEBUG_STRING
	DebugInterPredResiReco(sTest, *(m_ppcPredYuvTemp[uhDepth]), *(m_ppcResiYuvBest[uhDepth]), *(m_ppcRecoYuvTemp[uhDepth]), DebugStringGetPredModeMask(rpcTempCU->getPredictionMode(0)));
#endif

	xCheckDQP(rpcTempCU);
	xCheckBestMode(rpcBestCU, rpcTempCU, uhDepth DEBUG_STRING_PASS_INTO(sDebug) DEBUG_STRING_PASS_INTO(sTest));
}
// ֡��Ԥ�����ں���������ͨ�����ø�������ɾ����֡��Ԥ�⡢���롢����ģʽѡ����������а�������֡��Ԥ���ɫ��֡��Ԥ����������
Void TEncCu::xCheckRDCostIntra(TComDataCU*& rpcBestCU,
	TComDataCU*& rpcTempCU,
	Double& cost,
	PartSize     eSize
	DEBUG_STRING_FN_DECLARE(sDebug))
{
	DEBUG_STRING_NEW(sTest)
		if (getFastDeltaQp())
		{
			const TComSPS& sps = *(rpcTempCU->getSlice()->getSPS());
			const UInt fastDeltaQPCuMaxSize = Clip3(sps.getMaxCUHeight() >> (sps.getLog2DiffMaxMinCodingBlockSize()), sps.getMaxCUHeight(), 32u);
			if (rpcTempCU->getWidth(0) > fastDeltaQPCuMaxSize)
			{
				return; // only check necessary 2Nx2N Intra in fast deltaqp mode
			}
		}

	UInt uiDepth = rpcTempCU->getDepth(0);

	rpcTempCU->setSkipFlagSubParts(false, 0, uiDepth);

	rpcTempCU->setPartSizeSubParts(eSize, 0, uiDepth);
	rpcTempCU->setPredModeSubParts(MODE_INTRA, 0, uiDepth);
	rpcTempCU->setChromaQpAdjSubParts(rpcTempCU->getCUTransquantBypass(0) ? 0 : m_cuChromaQpOffsetIdxPlus1, 0, uiDepth);

	Pel resiLuma[NUMBER_OF_STORED_RESIDUAL_TYPES][MAX_CU_SIZE * MAX_CU_SIZE];
	// 帧内预测
	m_pcPredSearch->estIntraPredLumaQT(rpcTempCU, m_ppcOrigYuv[uiDepth], m_ppcPredYuvTemp[uiDepth], m_ppcResiYuvTemp[uiDepth], m_ppcRecoYuvTemp[uiDepth], resiLuma DEBUG_STRING_PASS_INTO(sTest));
	m_ppcRecoYuvTemp[uiDepth]->copyToPicComponent(COMPONENT_Y, rpcTempCU->getPic()->getPicYuvRec(), rpcTempCU->getCtuRsAddr(), rpcTempCU->getZorderIdxInCtu());

	if (rpcBestCU->getPic()->getChromaFormat() != CHROMA_400)
	{
		m_pcPredSearch->estIntraPredChromaQT(rpcTempCU, m_ppcOrigYuv[uiDepth], m_ppcPredYuvTemp[uiDepth], m_ppcResiYuvTemp[uiDepth], m_ppcRecoYuvTemp[uiDepth], resiLuma DEBUG_STRING_PASS_INTO(sTest));
	}

	m_pcEntropyCoder->resetBits();

	if (rpcTempCU->getSlice()->getPPS()->getTransquantBypassEnableFlag())
	{
		m_pcEntropyCoder->encodeCUTransquantBypassFlag(rpcTempCU, 0, true);
	}

	m_pcEntropyCoder->encodeSkipFlag(rpcTempCU, 0, true);
	m_pcEntropyCoder->encodePredMode(rpcTempCU, 0, true);
	m_pcEntropyCoder->encodePartSize(rpcTempCU, 0, uiDepth, true);
	m_pcEntropyCoder->encodePredInfo(rpcTempCU, 0);
	m_pcEntropyCoder->encodeIPCMInfo(rpcTempCU, 0, true);

	// Encode Coefficients
	Bool bCodeDQP = getdQPFlag();
	Bool codeChromaQpAdjFlag = getCodeChromaQpAdjFlag();
	m_pcEntropyCoder->encodeCoeff(rpcTempCU, 0, uiDepth, bCodeDQP, codeChromaQpAdjFlag);
	setCodeChromaQpAdjFlag(codeChromaQpAdjFlag);
	setdQPFlag(bCodeDQP);

	m_pcRDGoOnSbacCoder->store(m_pppcRDSbacCoder[uiDepth][CI_TEMP_BEST]);

	rpcTempCU->getTotalBits() = m_pcEntropyCoder->getNumberOfWrittenBits();
	rpcTempCU->getTotalBins() = ((TEncBinCABAC*)((TEncSbac*)m_pcEntropyCoder->m_pcEntropyCoderIf)->getEncBinIf())->getBinsCoded();
	rpcTempCU->getTotalCost() = m_pcRdCost->calcRdCost(rpcTempCU->getTotalBits(), rpcTempCU->getTotalDistortion());

	xCheckDQP(rpcTempCU);

	cost = rpcTempCU->getTotalCost();

	xCheckBestMode(rpcBestCU, rpcTempCU, uiDepth DEBUG_STRING_PASS_INTO(sDebug) DEBUG_STRING_PASS_INTO(sTest));
}


/** Check R-D costs for a CU with PCM mode.
 * \param rpcBestCU pointer to best mode CU data structure
 * \param rpcTempCU pointer to testing mode CU data structure
 * \returns Void
 *
 * \note Current PCM implementation encodes sample values in a lossless way. The distortion of PCM mode CUs are zero. PCM mode is selected if the best mode yields bits greater than that of PCM mode.
 */
Void TEncCu::xCheckIntraPCM(TComDataCU*& rpcBestCU, TComDataCU*& rpcTempCU)
{
	if (getFastDeltaQp())
	{
		const TComSPS& sps = *(rpcTempCU->getSlice()->getSPS());
		const UInt fastDeltaQPCuMaxPCMSize = Clip3((UInt)1 << sps.getPCMLog2MinSize(), (UInt)1 << sps.getPCMLog2MaxSize(), 32u);
		if (rpcTempCU->getWidth(0) > fastDeltaQPCuMaxPCMSize)
		{
			return;   // only check necessary PCM in fast deltaqp mode
		}
	}

	UInt uiDepth = rpcTempCU->getDepth(0);

	rpcTempCU->setSkipFlagSubParts(false, 0, uiDepth);

	rpcTempCU->setIPCMFlag(0, true);
	rpcTempCU->setIPCMFlagSubParts(true, 0, rpcTempCU->getDepth(0));
	rpcTempCU->setPartSizeSubParts(SIZE_2Nx2N, 0, uiDepth);
	rpcTempCU->setPredModeSubParts(MODE_INTRA, 0, uiDepth);
	rpcTempCU->setTrIdxSubParts(0, 0, uiDepth);
	rpcTempCU->setChromaQpAdjSubParts(rpcTempCU->getCUTransquantBypass(0) ? 0 : m_cuChromaQpOffsetIdxPlus1, 0, uiDepth);

	m_pcPredSearch->IPCMSearch(rpcTempCU, m_ppcOrigYuv[uiDepth], m_ppcPredYuvTemp[uiDepth], m_ppcResiYuvTemp[uiDepth], m_ppcRecoYuvTemp[uiDepth]);

	m_pcRDGoOnSbacCoder->load(m_pppcRDSbacCoder[uiDepth][CI_CURR_BEST]);

	m_pcEntropyCoder->resetBits();

	if (rpcTempCU->getSlice()->getPPS()->getTransquantBypassEnableFlag())
	{
		m_pcEntropyCoder->encodeCUTransquantBypassFlag(rpcTempCU, 0, true);
	}

	m_pcEntropyCoder->encodeSkipFlag(rpcTempCU, 0, true);
	m_pcEntropyCoder->encodePredMode(rpcTempCU, 0, true);
	m_pcEntropyCoder->encodePartSize(rpcTempCU, 0, uiDepth, true);
	m_pcEntropyCoder->encodeIPCMInfo(rpcTempCU, 0, true);

	m_pcRDGoOnSbacCoder->store(m_pppcRDSbacCoder[uiDepth][CI_TEMP_BEST]);

	rpcTempCU->getTotalBits() = m_pcEntropyCoder->getNumberOfWrittenBits();
	rpcTempCU->getTotalBins() = ((TEncBinCABAC*)((TEncSbac*)m_pcEntropyCoder->m_pcEntropyCoderIf)->getEncBinIf())->getBinsCoded();
	rpcTempCU->getTotalCost() = m_pcRdCost->calcRdCost(rpcTempCU->getTotalBits(), rpcTempCU->getTotalDistortion());

	xCheckDQP(rpcTempCU);
	DEBUG_STRING_NEW(a)
		DEBUG_STRING_NEW(b)
		xCheckBestMode(rpcBestCU, rpcTempCU, uiDepth DEBUG_STRING_PASS_INTO(a) DEBUG_STRING_PASS_INTO(b));
}

/** check whether current try is the best with identifying the depth of current try
 * \param rpcBestCU
 * \param rpcTempCU
 * \param uiDepth
 */
Void TEncCu::xCheckBestMode(TComDataCU*& rpcBestCU, TComDataCU*& rpcTempCU, UInt uiDepth DEBUG_STRING_FN_DECLARE(sParent) DEBUG_STRING_FN_DECLARE(sTest) DEBUG_STRING_PASS_INTO(Bool bAddSizeInfo))
{
	if (rpcTempCU->getTotalCost() < rpcBestCU->getTotalCost())
	{
		sflag = true;
		TComYuv* pcYuv;
		// Change Information data
		TComDataCU* pcCU = rpcBestCU;
		rpcBestCU = rpcTempCU;
		rpcTempCU = pcCU;

		// Change Prediction data
		pcYuv = m_ppcPredYuvBest[uiDepth];
		m_ppcPredYuvBest[uiDepth] = m_ppcPredYuvTemp[uiDepth];
		m_ppcPredYuvTemp[uiDepth] = pcYuv;

		// Change Reconstruction data
		pcYuv = m_ppcRecoYuvBest[uiDepth];
		m_ppcRecoYuvBest[uiDepth] = m_ppcRecoYuvTemp[uiDepth];
		m_ppcRecoYuvTemp[uiDepth] = pcYuv;

		pcYuv = NULL;
		pcCU = NULL;

		// store temp best CI for next CU coding
		m_pppcRDSbacCoder[uiDepth][CI_TEMP_BEST]->store(m_pppcRDSbacCoder[uiDepth][CI_NEXT_BEST]);


#if DEBUG_STRING
		DEBUG_STRING_SWAP(sParent, sTest)
			const PredMode predMode = rpcBestCU->getPredictionMode(0);
		if ((DebugOptionList::DebugString_Structure.getInt() & DebugStringGetPredModeMask(predMode)) && bAddSizeInfo)
		{
			std::stringstream ss(stringstream::out);
			ss << "###: " << (predMode == MODE_INTRA ? "Intra   " : "Inter   ") << partSizeToString[rpcBestCU->getPartitionSize(0)] << " CU at " << rpcBestCU->getCUPelX() << ", " << rpcBestCU->getCUPelY() << " width=" << UInt(rpcBestCU->getWidth(0)) << std::endl;
			sParent += ss.str();
		}
#endif
	}
}

Void TEncCu::xCheckDQP(TComDataCU* pcCU)
{
	UInt uiDepth = pcCU->getDepth(0);

	const TComPPS& pps = *(pcCU->getSlice()->getPPS());
	if (pps.getUseDQP() && uiDepth <= pps.getMaxCuDQPDepth())
	{
		if (pcCU->getQtRootCbf(0))
		{
			m_pcEntropyCoder->resetBits();
			m_pcEntropyCoder->encodeQP(pcCU, 0, false);
			pcCU->getTotalBits() += m_pcEntropyCoder->getNumberOfWrittenBits(); // dQP bits
			pcCU->getTotalBins() += ((TEncBinCABAC*)((TEncSbac*)m_pcEntropyCoder->m_pcEntropyCoderIf)->getEncBinIf())->getBinsCoded();
			pcCU->getTotalCost() = m_pcRdCost->calcRdCost(pcCU->getTotalBits(), pcCU->getTotalDistortion());
		}
		else
		{
			pcCU->setQPSubParts(pcCU->getRefQP(0), 0, uiDepth); // set QP to default QP
		}
	}
}

Void TEncCu::xCopyAMVPInfo(AMVPInfo* pSrc, AMVPInfo* pDst)
{
	pDst->iN = pSrc->iN;
	for (Int i = 0; i < pSrc->iN; i++)
	{
		pDst->m_acMvCand[i] = pSrc->m_acMvCand[i];
	}
}
Void TEncCu::xCopyYuv2Pic(TComPic* rpcPic, UInt uiCUAddr, UInt uiAbsPartIdx, UInt uiDepth, UInt uiSrcDepth)
{
	UInt uiAbsPartIdxInRaster = g_auiZscanToRaster[uiAbsPartIdx];
	UInt uiSrcBlkWidth = rpcPic->getNumPartInCtuWidth() >> (uiSrcDepth);
	UInt uiBlkWidth = rpcPic->getNumPartInCtuWidth() >> (uiDepth);
	UInt uiPartIdxX = ((uiAbsPartIdxInRaster % rpcPic->getNumPartInCtuWidth()) % uiSrcBlkWidth) / uiBlkWidth;
	UInt uiPartIdxY = ((uiAbsPartIdxInRaster / rpcPic->getNumPartInCtuWidth()) % uiSrcBlkWidth) / uiBlkWidth;
	UInt uiPartIdx = uiPartIdxY * (uiSrcBlkWidth / uiBlkWidth) + uiPartIdxX;
	m_ppcRecoYuvBest[uiSrcDepth]->copyToPicYuv(rpcPic->getPicYuvRec(), uiCUAddr, uiAbsPartIdx, uiDepth - uiSrcDepth, uiPartIdx);

	m_ppcPredYuvBest[uiSrcDepth]->copyToPicYuv(rpcPic->getPicYuvPred(), uiCUAddr, uiAbsPartIdx, uiDepth - uiSrcDepth, uiPartIdx);
}

Void TEncCu::xCopyYuv2Tmp(UInt uiPartUnitIdx, UInt uiNextDepth)
{
	UInt uiCurrDepth = uiNextDepth - 1;
	m_ppcRecoYuvBest[uiNextDepth]->copyToPartYuv(m_ppcRecoYuvTemp[uiCurrDepth], uiPartUnitIdx);
	m_ppcPredYuvBest[uiNextDepth]->copyToPartYuv(m_ppcPredYuvBest[uiCurrDepth], uiPartUnitIdx);
}

/** Function for filling the PCM buffer of a CU using its original sample array
 * \param pCU pointer to current CU
 * \param pOrgYuv pointer to original sample array
 */
Void TEncCu::xFillPCMBuffer(TComDataCU* pCU, TComYuv* pOrgYuv)
{
	const ChromaFormat format = pCU->getPic()->getChromaFormat();
	const UInt numberValidComponents = getNumberValidComponents(format);
	for (UInt componentIndex = 0; componentIndex < numberValidComponents; componentIndex++)
	{
		const ComponentID component = ComponentID(componentIndex);

		const UInt width = pCU->getWidth(0) >> getComponentScaleX(component, format);
		const UInt height = pCU->getHeight(0) >> getComponentScaleY(component, format);

		Pel* source = pOrgYuv->getAddr(component, 0, width);
		Pel* destination = pCU->getPCMSample(component);

		const UInt sourceStride = pOrgYuv->getStride(component);

		for (Int line = 0; line < height; line++)
		{
			for (Int column = 0; column < width; column++)
			{
				destination[column] = source[column];
			}

			source += sourceStride;
			destination += width;
		}
	}
}

#if ADAPTIVE_QP_SELECTION
/** Collect ARL statistics from one block
  */
Int TEncCu::xTuCollectARLStats(TCoeff* rpcCoeff, TCoeff* rpcArlCoeff, Int NumCoeffInCU, Double* cSum, UInt* numSamples)
{
	for (Int n = 0; n < NumCoeffInCU; n++)
	{
		TCoeff u = abs(rpcCoeff[n]);
		TCoeff absc = rpcArlCoeff[n];

		if (u != 0)
		{
			if (u < LEVEL_RANGE)
			{
				cSum[u] += (Double)absc;
				numSamples[u]++;
			}
			else
			{
				cSum[LEVEL_RANGE] += (Double)absc - (Double)(u << ARL_C_PRECISION);
				numSamples[LEVEL_RANGE]++;
			}
		}
	}

	return 0;
}

//! Collect ARL statistics from one CTU
Void TEncCu::xCtuCollectARLStats(TComDataCU* pCtu)
{
	Double cSum[LEVEL_RANGE + 1];     //: the sum of DCT coefficients corresponding to data type and quantization output
	UInt numSamples[LEVEL_RANGE + 1]; //: the number of coefficients corresponding to data type and quantization output

	TCoeff* pCoeffY = pCtu->getCoeff(COMPONENT_Y);
	TCoeff* pArlCoeffY = pCtu->getArlCoeff(COMPONENT_Y);
	const TComSPS& sps = *(pCtu->getSlice()->getSPS());

	const UInt uiMinCUWidth = sps.getMaxCUWidth() >> sps.getMaxTotalCUDepth(); // NOTE: ed - this is not the minimum CU width. It is the square-root of the number of coefficients per part.
	const UInt uiMinNumCoeffInCU = 1 << uiMinCUWidth;                          // NOTE: ed - what is this?

	memset(cSum, 0, sizeof(Double) * (LEVEL_RANGE + 1));
	memset(numSamples, 0, sizeof(UInt) * (LEVEL_RANGE + 1));

	// Collect stats to cSum[][] and numSamples[][]
	for (Int i = 0; i < pCtu->getTotalNumPart(); i++)
	{
		UInt uiTrIdx = pCtu->getTransformIdx(i);

		if (pCtu->isInter(i) && pCtu->getCbf(i, COMPONENT_Y, uiTrIdx))
		{
			xTuCollectARLStats(pCoeffY, pArlCoeffY, uiMinNumCoeffInCU, cSum, numSamples);
		}//Note that only InterY is processed. QP rounding is based on InterY data only.

		pCoeffY += uiMinNumCoeffInCU;
		pArlCoeffY += uiMinNumCoeffInCU;
	}

	for (Int u = 1; u < LEVEL_RANGE; u++)
	{
		m_pcTrQuant->getSliceSumC()[u] += cSum[u];
		m_pcTrQuant->getSliceNSamples()[u] += numSamples[u];
	}
	m_pcTrQuant->getSliceSumC()[LEVEL_RANGE] += cSum[LEVEL_RANGE];
	m_pcTrQuant->getSliceNSamples()[LEVEL_RANGE] += numSamples[LEVEL_RANGE];
}
//bool TEncCu::hytest(TComDataCU* pcCU, TComYuv* pcOrgYuv, TComYuv* pcPreYuv, int depthR, double tempRDCost)
//{
//	return false;
//	UInt uiPartSize = pcCU->getWidth(0);
//	if (uiPartSize != Size[1] || data0 || flag[1] == 2 || !data1)
//		return false;
//	UInt uiTrUnitIdx = 0, i = 0, j = 0;
//	const ComponentID compID = ComponentID(0);
//	const Int uiPartWidth = uiPartSize >> pcOrgYuv->getComponentScaleX(compID);
//	const Int uiPartHeight = uiPartSize >> pcOrgYuv->getComponentScaleY(compID);
//	vector<vector<int>> p(uiPartHeight, vector<int>(uiPartWidth));
//	const Pel* pSrc0 = pcOrgYuv->getAddr(compID, uiTrUnitIdx, uiPartWidth);
//	const Pel* pSrc1 = pcPreYuv->getAddr(compID, uiTrUnitIdx, uiPartWidth);
//
//	const Int  iSrc0Stride = pcOrgYuv->getStride(compID);
//	const Int  iSrc1Stride = pcPreYuv->getStride(compID);
//	long int n = uiPartWidth * uiPartHeight;
//	double s = 0, u = 0, s2 = 0, s3 = 0, s4 = 0, pd = 0, fd = 0, u1 = 0, Q1 = 0, u2 = 0, Q2 = 0, x = 0, x1 = 0, h = 0, v = 0;
//	double xu[2] = { 0 }, yu[2] = { 0 }, sx[2] = { 0 }, sy[2] = { 0 };
//	float r1;
//	bool flag1 = false;
//
//	//Output << uiPartHeight << "--" << uiPartWidth << "--" << uiPartSize << "--" << iSrc0Stride << "**" << iSrc1Stride << endl;
//	for (i = 0; i < uiPartHeight; i++)
//	{
//		for (j = 0; j < uiPartWidth; j++)
//		{
//			p[i][j] = pSrc0[j] - pSrc1[j];
//			s = s + p[i][j];
//		}
//		pSrc0 += iSrc0Stride;
//		pSrc1 += iSrc1Stride;
//	}
//
//
//	if (uiPartSize == Size[1])   // 如果当前迭代的块正是32CU
//	{
//		u = s / n;	// 残差平均值
//		int rsdTotal = 0;
//		s = 0;
//		// 统计总的残差值rsdTotal
//		for (i = 0; i < uiPartHeight; i++)
//		{
//			for (j = 0; j < uiPartWidth; j++)
//			{
//				s = s + (p[i][j] - u) * (p[i][j] - u);
//				rsdTotal += abs(p[i][j]);
//				//Output << p[i][j] << " ";
//			}
//			//Output << endl;
//		}
//
//		rsdAve = (double)rsdTotal / (double)(32 * 32);
//		s = s / (n - 1);
//		// 得到残差的二阶矩s2、三阶矩s3和四阶矩s4
//		for (i = 0; i < uiPartHeight; i++)
//		{
//			for (j = 0; j < uiPartWidth; j++)
//			{
//				s2 = s2 + pow((p[i][j] - u), 2);
//				s3 = s3 + pow((p[i][j] - u), 3);
//				s4 = s4 + pow((p[i][j] - u), 4);
//			}
//		}
//		s2 = s2 / (n - 1);	// 方差
//		rsdD = s2;
//
//		//rsdT[cnt32] = get_rsd(p, s2);
//
//		s3 = s3 / (n - 1);
//		s4 = s4 / (n - 1);
//		pd = s3 / pow(s2, 1.5);  // 偏度
//		fd = s4 / pow(s2, 2);    // 峰度
//		rsdF[0] = pd;
//		rsdF[1] = fd;
//		//rsdSF[cnt32] = subFD_32(p);
//		rsdND = calD(p);		// 局部方差和全局方差
//		rsdSAG = SAGD(p);		// 方向复杂度和方向复杂度方差
//		RDCost = tempRDCost;
//
//		u1 = 0;
//		Q1 = 0.0763227;
//		Q1 = sqrt(6.0 * (n - 2) / ((n + 1) * (n + 3)));
//		u2 = 3.0 - 6.0 / (n + 1);
//		Q2 = 0.0230969;
//		Q2 = sqrt(24.0 * n * (n - 2) * (n - 3) / ((n + 1) * (n + 1) * (n + 3) * (n + 5)));
//
//		rsdU[0] = fabs(pd) / Q1;
//		rsdU[1] = fabs(fd - u2) / Q2;
//
//		if (depthR == 0)
//		{
//			r1 = 0.49;
//			if ((fabs(pd) / Q1 < r1) && (fabs(fd - u2) / Q2 < r1))
//			{
//				flag1 = true;
//			}
//		}
//		if (depthR == 1)
//		{
//
//			r1 = 2.61;
//			if ((fabs(pd) / Q1 < r1) && (fabs(fd - u2) / Q2 < r1))
//			{
//				flag1 = true;
//			}
//		}
//		if (depthR == 2)
//		{
//			r1 = 2.61;
//			if ((fabs(pd) / Q1 < r1) && (fabs(fd - u2) / Q2 < r1))
//			{
//				flag1 = true;
//			}
//		}
//		tpSample.clear();
//		tpSample.push_back(log10(rsdU[0]));
//		tpSample.push_back(log10(rsdU[1]));
//		tpSample.push_back(log10(rsdND[0]));
//		tpSample.push_back(log10(rsdND[1]));
//		tpSample.push_back(log10(1.0 * qp));
//		tpSample.push_back(RDCost / 10000.0);
//		sample1R.push_back(tpSample);
//	}
//
//	//if (uiPartSize == 32 && cnt32 == PARTS)
//	//{
//	//    for (i = 0; i < 4; i++)
//	//    {
//	//        RF << log10(rsdU[0]) << "\t" << log10(rsdU[1]) << "\t";
//	//        RF << log10(rsdND[0]) << "\t" << log10(rsdND[1]) << "\t" << qp << "\t";	// 局部方差和全局方差
//	//        RF << log10(rsdSAG[0]) << "\t" << log10(rsdSAG[1]) << "\t";	// 方向复杂度、方向复杂度方差
//	//        RF << rsdAve << "\t";	// 均值
//	//        RF << RDCost / 10000.0;
//	//        RF << endl;
//	//    }
//	//}
//	//u = s / n;
//	//s = 0;
//	//for (i = 0; i < uiPartHeight; i++)
//	//{
//	//    for (j = 0; j < uiPartWidth; j++)
//	//    {
//	//        s = s + (p[i][j] - u) * (p[i][j] - u);
//	//    }
//	//}
//	//s = s / (n - 1);
//	//for (i = 0; i < uiPartHeight; i++)
//	//{
//	//    for (j = 0; j < uiPartWidth; j++)
//	//    {
//	//        s2 = s2 + pow((p[i][j] - u), 2);
//	//        s3 = s3 + pow((p[i][j] - u), 3);
//	//        s4 = s4 + pow((p[i][j] - u), 4);
//	//    }
//	//}
//	//s2 = s2 / (n - 1);
//	//s3 = s3 / (n - 1);
//	//s4 = s4 / (n - 1);
//	//pd = s3 / pow(s2, 1.5);  // 偏度
//	//fd = s4 / pow(s2, 2);    // 峰度
//	//u1 = 0;
//	//Q1 = sqrt(6.0 * (n - 2) / ((n + 1) * (n + 3)));
//	//u2 = 3.0 - 6.0 / (n + 1);
//	//Q2 = sqrt(24.0 * n * (n - 2) * (n - 3) / ((n + 1) * (n + 1) * (n + 3) * (n + 5)));
//	//if (depthR == 0)
//	//{
//	//    r1 = 0.49;
//	//    if ((fabs(pd) / Q1 < r1) && (fabs(fd - u2) / Q2 < r1))
//	//    {
//	//        flag1 = true;
//	//    }
//	//}
//	//if (depthR == 1)
//	//{
//	//    r1 = 1.96;
//	//    if ((fabs(pd) / Q1 < r1) && (fabs(fd - u2) / Q2 < r1))
//	//    {
//	//        flag1 = true;
//	//    }
//	//}
//	//if (depthR == 2)
//	//{
//	//    r1 = 2.61;
//	//    if ((fabs(pd) / Q1 < r1) && (fabs(fd - u2) / Q2 < r1))
//	//    {
//	//        flag1 = true;
//	//    }
//	//}
//	return flag1;
//}


#endif
//! \}
