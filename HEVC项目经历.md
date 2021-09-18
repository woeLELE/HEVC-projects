# 一、项目描述

在视频编码标准HEVC/h.265中，每一帧都是由若干个编码树单元（CTU）组成的，CTU的大小通常为64*64，而每个CTU又可以划分为若干个编码单元（CU），划分方式为四叉树，CU形状也是方形，当CTU边长为64时，CU的大小可能为64、32、16和8，对应划分深度分别为0、1、2和3。为了找出最优的划分方式，原始HEVC的编码过程是编码所有大小的CU，从而遍历所有的划分方式，并找出具有最小率失真代价（RDCost）的划分方式， 这个过程是HEVC编码复杂度的主要来源，使得HEVC的编码时间效率较低。在此项目中，通过使用机器学习工具Libinear进行在线学习，对CU的灰度图以及预测编码结束生成的残差图进行分析，进行二分类，进而对划分方式进行提前决策，从而提前判断某个CU是继续划分或是提前终止，大大提高了编码效率。

# 二、项目流程

## 可行性验证

为验证可行性，我新建了一个机器学习项目，该项目通过文件读入数据集。在这个项目中，我掌握了Liblinear的使用方法，设计并模拟了在线学习的过程，验证了可行性。此学习过程中，还遇到了很多问题。

1. 训练集中正类和负类的样本个数通常相差较大，这往往会降低模型的准确性，所以我使用了一个方法实现了均等化，可以使得正类和负类的样本个数最终相差不会超过1，而且整个数据集中前后任意位置的样本都有机会被选取为训练集。实现代码如下：

   ```cpp
   // temp1和temp2分别为当前训练集中一类和二类样本的数量
   if (a > DEPTH && temp2 - temp1 <= 1 && temp2 - temp1 >= 0)	// 如果属于一类
   {
       vector<double> rx;
       temp1++;
       for (int i = 0; i < DIM; i++)
           rx.push_back(b[i]);	// 生成一类样本参数
       x.push_back(rx);
       y.push_back(1);	// 生成样本标签
   }
   else if (a == DEPTH && temp1 - temp2 <= 1 && temp1 - temp2 >= 0)	// 如果属于二类
   {
       vector<double> rx;
       temp2++;
       for (int i = 0; i < DIM; i++)
           rx.push_back(b[i]);
       x.push_back(rx);
       y.push_back(0);
   }
   ```
   
2. Liblinear的训练和预测效率很高，但是预测效果不佳，查准率通常无法满足要求，为了提高正类或者负类的查准率，需要设置它们的权值，如果提高某一类的权值，则此类的查全率会提高，而查准率会降低，另一类的查全率会降低，查准率会提高，反之亦然。而在这里，查准率更加重要，并且因为Liblinear预测效果的限制，往往只能保证某一类的查准率高于90%。
    为了设置适当的权值，这里我采用了自适应的方法。经过研究发现，数据集中某一类的查准率和该类在数据集中的占比有很大关系。所以我创建了一个函数，可以根据总数据集中正类和负类的比例判断是进行正类预测还是负类预测，进一步地，又创建一个函数，可以根据总数据集中正类和负类的比值、当前CU的深度以及预测类型（纹理/残差）自适应地调整权值。

3. 另外一个问题是训练集的选择问题，假设一个序列有100帧，那么大约有两种在线预测方式，第一种是选择固定的前面的几帧作为训练帧，生成训练集以及模型，并对后面帧的划分过程进行预测。另一种是动态地调整训练集和模型，比如选择初始的0、1、2共三帧图像作为训练帧，生成模型预测第3帧，第3帧预测完成后，将这一帧的预测结果作为标签和特征值合为样本，并和第1、2帧重新组合成训练集，生成模型预测第4帧，以此类推。

前一种方式的好处显而易见，实现容易，但是对比较靠后的帧可能预测效果不好。后一种方式实现比较困难，每编码完一帧都需要更新训练集并生成模型，而且存在错误传播，因为只有初始的几帧是使用原始的编码过程，后面的都存在误差。两种方式都实现以后，发现第一种效果更好。

## 在编码器内部实现在线学习

首先需要把Liblinear嵌入编码器。因为HEVC的标准参考软件HM编码器和Liblinear框架都是基于C++编写的，所以移植过程没有遇到太多困难。下面展示主要代码。

### 1. FUNCs

新建了FUNCs头文件和实现类，用来存放重要的变量函数。

```cpp
const int CTUS_LIMIT = 400;
const int PARTS1 = 4;
const int PARTS2 = 16;

// 根据传入的灰度图或残差图计算特征值
vector<double> calD(vector<vector<int>>& p);		// 计算局部方差以及平均方差
vector<double> SAGD(vector<vector<int>>& p);		// 计算方向复杂度以及方向复杂度方差

bool IsN(int a, int depth);	// 如果返回true，则为负类
int JudgeClass(int& cnt1, int& cnt0, int depth);	// 判断类别，因为某个深度可能进行正类预测（即使得正类查准率满足要求），也可能进行负类预测，还可能干脆不预测
CxLibLinear CreateModel(vector<vector<double>>& x, vector<double>& y, int& cnt1, int& cnt0, int label, int demand, char TorR, int D, int qp);	// 生成模型
vector<double> SetWeights(int cnt1, int cnt0, int label, int demand, char TorR, int D, int qp);	// 根据训练集中正类和负类的样本数确定权值比
CxLibLinear GetPredictor(double weight1, double weight0);		// 得到一个预测器

```

#### calD

```cpp
vector<double> calD(vector<vector<int>>& p)	// 计算局部方差以及平均方差
{
	int l = p.size();
	int size = l * l;
	vector<double> res(2);
	vector<vector<double>> mg(l, vector<double>(l));
	int pos[8][2] = { {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1} };	// 与某一像素相邻的各像素相对位置
	for (int i = 0; i < l; i++)
	{
		for (int j = 0; j < l; j++)
		{
			int cnt = 0;	// 在范围内的相邻像素的数量
			double nbt = 0;	// 相邻像素的灰度值总和
			for (int k = 0; k < 8; k++)
			{
				int x = i + pos[k][0];	// 相邻像素的横坐标
				int y = j + pos[k][1];	// 相邻像素的纵坐标
				if (x >= 0 && x < l && y >= 0 && y < l)
				{
					nbt += p[x][y];
					cnt++;
				}
			}	// 计算完成满足条件的i,j位置的相邻像素的数量以及相邻像素的灰度值总和
			mg[i][j] = nbt / (double)cnt;
		}
	}
	// 利用p和mg来计算neighboring mean squared error(NMSE)
	double va = 0.0;    // 记录差平方的总和

	int cnt1 = 0;
	int cnt2 = 0;
	int temp_gray = 0;        // 用来临时记录各个CU中的灰度值的总和
	double temp = 0.0;  // 临时记录差平方的和

	for (int i = 0; i < l; i++)
	{
		for (int j = 0; j < l; j++)
		{
			// 处理局部方差
			temp_gray += p[i][j];
			double temp = ((double)(p[i][j]) - mg[i][j]) * ((double)(p[i][j]) - mg[i][j]);
			va += temp;
		}
	}

	double ave_gray = (double)temp_gray / (double)size;		// 该CU中的平均灰度值
	for (int i = 0; i < l; i++)
	{
		for (int j = 0; j < l; j++)
		{
			// 处理平均方差
			double td = ((double)(p[i][j]) - ave_gray);
			temp += td * td;
		}
	}
	res[0] = (va / (double)size) <= 0 ? -100 : log10(va / (double)size);
	res[1] = (temp / (double)size) <= 0 ? -100 : log10(temp / (double)size);
	return res;
}
```

#### SAGD

```cpp
vector<double> SAGD(vector<vector<int>>& p)		// 计算方向复杂度以及方向复杂度方差
{
	int l = p.size();
	int size = l * l;
	int subl = l / 2;
	int subsize = subl * subl;
	vector<double> res(2);

	int p12[4][4] = { {0, -1, 0, 1}, {-1, 0, 1, 0}, {-1, -1, 1, 1}, {-1, 1, 1, -1} };	// 第一个维度为四个方向，第二个维度为该方向对应的两个像素相对位置坐标

	vector<vector<vector<int>>> dis(4, vector<vector<int>>(l, vector<int>(l)));			// 四个方向上的每个像素点的梯度
	for (int m = 0; m < 4; m++)
	{
		for (int i = 0; i < l; i++)
		{
			for (int j = 0; j < l; j++)
			{
				int x1 = i + p12[m][0];
				int y1 = j + p12[m][1];
				int x2 = i + p12[m][2];
				int y2 = j + p12[m][3];
				int tempDis;
				if (x1 >= 0 && x1 < l && y1 >= 0 && y1 < l && x2 >= 0 && x2 < l && y2 >= 0 && y2 < l)
					tempDis = p[x1][y1] - p[x2][y2];
				else
					tempDis = 0;

				if (tempDis < 0)
					tempDis = -tempDis;
				dis[m][i][j] = tempDis;
			}
		}
	}	// 得到每个像素在四个方向下的梯度

	int subpos[4][2] = { {0, 0}, {0, subl}, {subl, 0}, {subl, subl} };	// 各个子CU在其CU中的相对起始位置
	double tpSubS[4] = { 0 };	// 存储各个CU的方向复杂度
	// 求各个CU及CU的方向复杂度
	int totalDis = 0;			// 存储当前CU的每个像素的梯度之和

	for (int i = 0; i < l; i++)
		for (int j = 0; j < l; j++)
			for (int n = 0; n < 4; n++)
				totalDis += dis[n][i][j];

	res[0] = (double)totalDis / size <= 0 ? -100 : log10((double)totalDis / size);

	for (int n = 0; n < 4; n++)	// 在各个子CU中
	{
		int p1 = subpos[n][0];	// 当前子CU在整个当前CU中的起始位置
		int p2 = subpos[n][1];
		for (int i = p1; i < p1 + subl; i++)
			for (int j = p2; j < p2 + subl; j++)
				for (int k = 0; k < 4; k++)
					tpSubS[n] += (double)dis[k][i][j];	// 先得到每个子CU中像素的梯度总和
		tpSubS[n] /= (double)subsize;
	}

	double temp = 0;
	for (int j = 0; j < 4; j++)	// 对于各个子CU
		temp += tpSubS[j];
	temp /= 4.0;	// 代表当前CU中各个子CU的方向复杂度的均值

	double total = 0;
	for (int j = 0; j < 4; j++)
		total += (tpSubS[j] - temp) * (tpSubS[j] - temp);
	res[1] = total / 4.0 <= 0 ? -100 : log10(total / 4.0);
	return res;
}
```

#### JudgeClass

```cpp
// 判断各个深度的类别
int JudgeClass(int& cnt1, int& cnt0, int depth)
{
	if (depth == 0 && cnt1 > cnt0)
		return 1;
	if (depth == 1)
	{
		if (cnt1 > cnt0)
			return 1;
		else if (cnt0 > cnt1)
			return 2;
	}
	if (depth == 2 && cnt1 <= cnt0)
		return 1;
	else if (depth == 2 && cnt1 > cnt0)
		return 2;
	return 0;
}
```

#### CreateModel

```cpp
// 生成一个模型，其中调用了其他相关函数
CxLibLinear CreateModel(vector<vector<double>>& x, vector<double>& y, int& cnt1, int& cnt0, int label, int demand, char TorR, int D, int qp)
{
	vector<double> weight = SetWeights(cnt1, cnt0, label, demand, TorR, D, qp);	// 设置权值
	CxLibLinear linear = GetPredictor(weight[0], weight[1]);    // 得到预测器
	linear.train_model(x, y);   // 训练模型
	return linear;
}
```

#### GetPredictor

```cpp
// 根据传入的权值得到预测器
CxLibLinear GetPredictor(double weight1, double weight0)
{
	CxLibLinear linear;
	parameter param;
	param.solver_type = L2R_LR;
	param.eps = 0.001;	/* stopping criteria 停止迭代的标准（精度）*/
	param.C = 2;			/*违法约束后的代价*/
	param.nr_weight = 2;	/*用于改变某些类别的惩罚，如何你不想改变任何类型的惩罚，只要把nr_weight设置为0。*/
	param.weight_label = Malloc(int, 2);
	param.weight_label[0] = 1;
	param.weight_label[1] = 0;
	param.weight = Malloc(double, 2);
	param.weight[0] = weight1;
	param.weight[1] = weight0;
	param.p = 0.001;			/*支持向量回归损失的灵敏度*/
	param.init_sol = NULL;

	linear.init_linear_param(param);
	return linear;
}
```

### 2. 生成训练集

在视频编码标准HEVC/h.265中，每一帧都是由若干个编码树单元（CTU）组成的，CTU的大小通常为64*64，而每个CTU又可以划分为若干个编码单元（CU），划分方式为四叉树，CU形状也是方形，当CTU边长为64时，CU的大小可能为64、32、16和8，对应划分深度分别为0、1、2和3。
负责预测编码和CU划分的函数如下所示。

```cpp
Void TEncCu::compressCtu(TComDataCU* pCtu)  // 编码一个CTU
Void TEncCu::xCompressCU(TComDataCU*& rpcBestCU, TComDataCU*& rpcTempCU, const UInt uiDepth DEBUG_STRING_FN_DECLARE(sDebug_), PartSize eParentPartSize) // 编码一个CU
```

原始编码器中，compressCtu的实现很简单，就是调用一下xCompressCU。
在compressCtu中，调用xCompressCU的方式如下：

```cpp
xCompressCU(m_ppcBestCU[0], m_ppcTempCU[0], 0 DEBUG_STRING_PASS_INTO(sDebug));  // uiDepth设置为0，即为从深度0（也就是最大的CU）开始编码CU
```

在xCompressCU中，如果当前CU深度小于3，又会循环递归自己4次：

```cpp
// 原本代码要复杂很多，这里只是展示一下递归方式
for (UInt uiPartUnitIdx = 0; uiPartUnitIdx < 4; uiPartUnitIdx++)
    xCompressCU(pcSubBestPartCU, pcSubTempPartCU, uhNextDepth DEBUG_STRING_PASS_INTO(sChild), NUMBER_OF_PART_SIZES);    // uhNextDepth即为下一个深度
```

有了以上说明，接下来更细致地说明原始的CU编码过程。
假设当前CU深度为0，则为了得到当前CU的**最优/最小率失真代价（RDCost）**，需要进行如下过程（同时也是xCompressCU函数的执行过程）：

1. 编码整个CU，得到其RDCost，记为C1；

2. 编码其中4个深度为1的子CU，得到各个子CU的**最优**RDCost，并计算它们的和，记为C2；

3. 比较C1和C2，取最小值为最优RDCost。如果C1更小，则当前CU不划分，反之则继续划分。

而为了得到每个深度为1的子CU的最优RDCost，又需要重复以上过程，直到当前CU深度为3。
也就是说，不管最终划分方式如何，为了找到最优的划分方式，原始编码器总是会对所有大小的CU进行编码，这导致了巨大的计算量。
而在上述第1步之前，可以得到当前CU的灰度图，如果根据灰度图得到的特征值可以提前确定当前CU继续划分，则可以跳过第1步；如果可以提前确定当前CU终止划分，那就更棒了，可以跳过第2步，后面的递归都不用做了。
因为灰度图和残差图都是在xCompressCU中得到的，所以特征值也可以在这里得到，但是只有当`xCompressCU(m_ppcBestCU[0], m_ppcTempCU[0], 0 DEBUG_STRING_PASS_INTO(sDebug));`执行完以后，再次返回到compressCtu中时，才能得到最终的划分方式，才能得到标签，而训练集中的样本必须同时包含特征值和标签，所以得到训练集以及生成模型的步骤只能放在compressCtu中了。

以下为compressCtu中全部代码，在生成各个深度训练集的过程中，使用了**均等化**的方法
```cpp
Void TEncCu::compressCtu(TComDataCU* pCtu)
{
	isComplete = true;  // 用以判断当前CTU是否时完整的，如果不是完整的，则使用原始的编码过程
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
		if (pCtu->getSlice()->getSliceType() != I_SLICE)
		{
			xCtuCollectARLStats(pCtu);
		}
	}
#endif
}
```

### 3. 进行预测

在xCompressCU中，可以根据灰度图或残差图得到特征值并进行预测。

以下为xCompressCU中的部分代码，展示了得到灰度图并生成纹理特征值和进行纹理预测的部分
```cpp
Void TEncCu::xCompressCU(TComDataCU*& rpcBestCU, TComDataCU*& rpcTempCU, const UInt uiDepth DEBUG_STRING_FN_DECLARE(sDebug_), PartSize eParentPartSize)
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
			if (!data0)	// 如果还没有生成完整的训练集以及模型
			{
				if (depth == 0)
					sample0 = tpSample;
				else if (depth == 1)
					sample1T.push_back(tpSample);
				else if (depth == 2)
					sample2.push_back(tpSample);
			}
			else if (flag[depth] > 0)	// 否则进行预测
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

	split = split || (pre0 == 1 || pre1_1 == 1 || pre1_2 == 1 || pre2 == 1);    // 得知当前CU是否继续划分
	cease = cease || (pre0 == 0 || pre1_1 == 0 || pre1_2 == 0 || pre2 == 0);    // 得知当前CU是否终止划分
```
后面主要是HEVC原始的编码过程，就不展示了。当然，我在原始编码器中增加的代码还不止于此，在某些情况下，将进一步进行残差预测；当doRMD设置为true时，会进一步地进行帧内模式的提前决策；此外，深度3的CU大小为8*8，这也是最小的CU尺寸，但是，在帧内预测模式下，它可以继续划分为4个预测单元（PU），此划分过程有着和CU划分相似的规律，对此我也做了提前决策，限于篇幅，就不展示了。

## 三、成果与总结
在几乎不降低视觉质量和不提高比特率的前提下HEVC编码时间效率提高45%。

这是我做的第一个项目，并且独立完成了代码的流程设计与编写。刚开始我连vector是啥都不知道，而在实现这个项目的过程中我的各项能力得到了很大的提高。
