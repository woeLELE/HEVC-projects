#include "FUNCs.h"

#define DIMT 5
#define DIMR 6


//ofstream Tout("Tfile2.txt");
//ofstream Rout("G:\\HEVC\\������\\���\\�в����ݼ�.txt");
//ofstream Lout("G:\\HEVC\\������\\���\\���.txt");


//ofstream Out("G:\\HEVC\\������\\���\\FUNCs���Խ��.txt");
ofstream Out("FUNCs���Խ��.txt");

double** subD(int(*p)[64])	// ����ÿ��32CU�µ��ĸ��ӿ�֮��������Ӷȷ���
{
	double** res = new double* [4];
	for (int i = 0; i < 4; i++)
		res[i] = new double[2];
	int templ = 32;
	double s = 0, v2 = 0, s3 = 0, s4 = 0, u = 0;
	double fc = 0, pd = 0, fd = 0;
	int sp[4][2] = { {0, 0}, {0, templ}, {templ, 0}, {templ, templ} };   // ��CU���������������
	int m, n, i, j, r1;
	for (r1 = 0; r1 < 4; r1++)		// ����CTU�е��ĸ�32CU
	{
		double a[4][2];		// �����洢��ǰ32CU�е��ĸ��ӿ���Ե�3����������ֵ�����Լ�������֮��ķ���
		int x = sp[r1][0];	// ��ǰCU��CTU�����������
		int y = sp[r1][1];	// ��ǰCU��CTU�����������
		int k = 0;	// ��k��16CU
		double total[2] = { 0 };	// �洢�ĸ�16���2������ֵ���Ե��ܺ�
		for (i = x; i < x + templ; i = i + templ / 2)
		{
			for (j = y; j < y + templ; j = j + templ / 2)
			{
				s = 0;
				fc = 0;
				v2 = 0;
				s3 = 0;
				s4 = 0;
				for (m = i; m < i + templ / 2; m++)
				{
					for (n = j; n < j + templ / 2; n++)
					{
						s += p[m][n];
					}
				}
				u = s / pow(16, 2);
				for (m = i; m < i + templ / 2; m++)
				{
					for (n = j; n < j + templ / 2; n++)
					{
						v2 = v2 + pow((p[m][n] - u), 2);
						s3 = s3 + pow((p[m][n] - u), 3);
						s4 = s4 + pow((p[m][n] - u), 4);
					}
				}
				n = 16 * 16;
				fc = v2 / (n - 1);
				v2 = v2 / n;
				s3 = s3 / n;
				s4 = s4 / n;
				pd = s3 / pow(v2, 1.5);
				fd = s4 / pow(v2, 2);
				//}
				a[k][0] = log10(fc);
				//a[k][1] = pd;
				a[k][1] = log10(fd);

				for (int t = 0; t < 2; t++)		// ����k��16�������ֵ�����ܺ�
					total[t] += a[k][t];
				k++;
			}
		}	// �������һ��32CU�е��ĸ��ӿ���Ե�����ֵ
		double ave[2];	// ����ֵ��ƽ��ֵ
		for (i = 0; i < 2; i++)
			ave[i] = total[i] / 4;

		for (i = 0; i < 2; i++)		// ����2������ֵ
		{
			double temptotal = 0;	// �洢��ƽ��֮��
			for (j = 0; j < 4; j++)
			{
				double temp = a[j][i] - ave[i];
				//if (temp < 0)
				//	temp = -temp;
				temptotal += temp * temp;
			}
			res[r1][i] = temptotal / 4;
		}	// �������һ��32CU�е���������ֵ���Եķ���
		//res.push_back(tempres);
	}
	return res;
}

vector<double> subFD_32(vector<vector<int>> p)	// ����ÿ��32CU�µ��ĸ��ӿ�֮��������Ӷȷ���
{
	//vector<vector<double>> res;		// �洢�ĸ�32CU���Ե������Ӷȷ���
	vector<double> res(3);
	int templ = 32;
	double s = 0, v2 = 0, s3 = 0, s4 = 0, u = 0;
	double fc = 0, pd = 0, fd = 0;
	int m, n, i, j, r1;

	double a[4][3];		// �����洢��ǰ32CU�е��ĸ��ӿ���Ե�3����������ֵ�����Լ�������֮��ķ���
	int k = 0;	// ��k��16CU
	double total[3] = { 0 };	// �洢�ĸ�16���3������ֵ���Ե��ܺ�
	for (i = 0; i < templ; i = i + templ / 2)
	{
		for (j = 0; j < templ; j = j + templ / 2)
		{
			s = 0;
			fc = 0;
			v2 = 0;
			s3 = 0;
			s4 = 0;
			for (m = i; m < i + templ / 2; m++)
			{
				for (n = j; n < j + templ / 2; n++)
				{
					s += p[m][n];
				}
			}
			u = s / pow(32, 2);
			for (m = i; m < i + templ / 2; m++)
			{
				for (n = j; n < j + templ / 2; n++)
				{
					v2 = v2 + pow((p[m][n] - u), 2);
					s3 = s3 + pow((p[m][n] - u), 3);
					s4 = s4 + pow((p[m][n] - u), 4);
				}
			}
			fc = v2 / (n - 1);
			v2 = v2 / n;
			s3 = s3 / n;
			s4 = s4 / n;
			if (v2 < 1)
			{
				fc = 1;
				pd = 1;
				fd = 1;
			}
			else
			{
				pd = s3 / pow(v2, 1.5);
				fd = s4 / pow(v2, 2);
			}
			a[k][0] = log10(fc);
			a[k][1] = pd;
			a[k][2] = log10(fd);

			for (int t = 0; t < 3; t++)		// ����k��16�������ֵ�����ܺ�
				total[t] += a[k][t];
			k++;
		}
	}	// �������һ��32CU�е��ĸ��ӿ���Ե�����ֵ
	double ave[3];	// ����ֵ��ƽ��ֵ
	for (i = 0; i < 3; i++)
		ave[i] = total[i] / 4;

	/*for (i = 0; i < 3; i++)
	{
		Out << total[i] << "---" << ave[i] << " ";
	}
	Out << endl;*/
	for (i = 0; i < 3; i++)		// ����3������ֵ
	{
		double temptotal;	// �洢��ƽ��֮��
		for (j = 0; j < 4; j++)
		{
			double temp = a[j][i] - ave[i];
			//if (temp < 0)
			//	temp = -temp;
			temptotal += temp * temp;
		}
		res[i] = temptotal / 4;
	}	// �������һ��32CU�е���������ֵ���Եķ���
	//res.push_back(tempres);

	return res;
}

vector<double> calD(vector<vector<int>>& p)	// ����ֲ������Լ�ƽ������
{
	int l = p.size();
	int size = l * l;
	vector<double> res(2);
	vector<vector<double>> mg(l, vector<double>(l));
	int pos[8][2] = { {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1} };	// ��ĳһ�������ڵĸ��������λ��
	for (int i = 0; i < l; i++)
	{
		for (int j = 0; j < l; j++)
		{
			int cnt = 0;	// �ڷ�Χ�ڵ��������ص�����
			double nbt = 0;	// �������صĻҶ�ֵ�ܺ�
			for (int k = 0; k < 8; k++)
			{
				int x = i + pos[k][0];	// �������صĺ�����
				int y = j + pos[k][1];	// �������ص�������
				if (x >= 0 && x < l && y >= 0 && y < l)	// ��������������ڷ�Χ��
				{
					nbt += p[x][y];
					cnt++;
				}
			}	// �����������������i,jλ�õ��������ص������Լ��������صĻҶ�ֵ�ܺ�
			mg[i][j] = nbt / (double)cnt;
		}
		//mg.push_back(temp);
	}

	// ����p��mg������neighboring mean squared error(NMSE)
	double va = 0;    // ��¼��ƽ�����ܺ�
	int cnt1 = 0;
	int cnt2 = 0;
	int temp_gray = 0;        // ������ʱ��¼����CU�еĻҶ�ֵ���ܺ�
	double temp = 0;  // ��ʱ��¼��ƽ���ĺ�

	for (int i = 0; i < l; i++)
	{
		for (int j = 0; j < l; j++)
		{
			// ����ֲ�����
			temp_gray += p[i][j];
			double temp = (double(p[i][j]) - mg[i][j]) * (double(p[i][j]) - mg[i][j]);
			va += temp;

		}
	}

	double ave_gray = double(temp_gray) / double(size);		// ��CU�е�ƽ���Ҷ�ֵ
	for (int i = 0; i < l; i++)
	{
		for (int j = 0; j < l; j++)
		{
			// ����ƽ������
			double td = ((double)(p[i][j]) - ave_gray);
			temp += td * td;
		}
	}
	res[0] = (va / (double)size) <= 0 ? -100 : log10(va / (double)size);
	res[1] = (temp / (double)size) <= 0 ? -100 : log10(temp / (double)size);
	return res;
}

vector<double> SAGD(vector<vector<int>>& p)		// ���㷽���Ӷ��Լ������Ӷȷ���
{
	int l = p.size();
	int size = l * l;
	int subl = l / 2;
	int subsize = subl * subl;
	vector<double> res(2);

	int p12[4][4] = { {0, -1, 0, 1}, {-1, 0, 1, 0}, {-1, -1, 1, 1}, {-1, 1, 1, -1} };	// ��һ��ά��Ϊ�ĸ����򣬵ڶ���ά��Ϊ�÷����Ӧ�������������λ������

	vector<vector<vector<int>>> dis(4, vector<vector<int>>(l, vector<int>(l)));			// �ĸ������ϵ�ÿ�����ص���ݶ�
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
				{
					tempDis = p[x1][y1] - p[x2][y2];
				}
				else
					tempDis = 0;

				if (tempDis < 0)
					tempDis = -tempDis;
				dis[m][i][j] = tempDis;
				//Out << tempDis << " ";
			}
			//Out << endl;
		}
	}	// �õ�ÿ���������ĸ������µ��ݶ�

	int subpos[4][2] = { {0, 0}, {0, subl}, {subl, 0}, {subl, subl} };	// ������CU����CU�е������ʼλ��
	double tpSubS[4] = { 0 };	// �洢����CU�ķ����Ӷ�
	// �����CU��CU�ķ����Ӷ�
	int totalDis = 0;			// �洢��ǰCU��ÿ�����ص��ݶ�֮��

	for (int i = 0; i < l; i++)
		for (int j = 0; j < l; j++)	// ����ÿ������
			for (int n = 0; n < 4; n++)	// ����ÿ�����ص��ĸ�����
				totalDis += dis[n][i][j];

	res[0] = (double)totalDis / size <= 0 ? -100 : log10((double)totalDis / size);

	for (int n = 0; n < 4; n++)	// �ڸ�����CU��
	{
		int p1 = subpos[n][0];	// ��CU����ʼλ��
		int p2 = subpos[n][1];
		for (int i = p1; i < p1 + subl; i++)
			for (int j = p2; j < p2 + subl; j++)
				for (int k = 0; k < 4; k++)
					tpSubS[n] += (double)dis[k][i][j];	// �ȵõ�ÿ����CU�����ص��ݶ��ܺ�
		tpSubS[n] /= (double)subsize;
	}

	double temp = 0;
	for (int j = 0; j < 4; j++)	// ���ڸ���16CU
		temp += tpSubS[j];
	temp /= 4.0;	// ����ǰ32CU�и�����CU�ķ����Ӷȵľ�ֵ

	double total = 0;
	for (int j = 0; j < 4; j++)
	{
		total += (tpSubS[j] - temp) * (tpSubS[j] - temp);
	}
	res[1] = total / 4.0 <= 0 ? -100 : log10(total / 4.0);
	return res;
}

double ave_D(vector<vector<vector<int>>>& p, vector<int>& subAve)
{
	//for (auto st : subAve)
	//	Out << st << " ";
	//Out << endl;
	int h = p.front().size();
	int w = p.front().front().size();
	//double tD = 0.0;
	int cnt = 0;
	double tD = 0.0;
	for (auto s : p)
	{
		double total = 0.0;
		for (auto v : s)
			for (auto n : v)
				total += (n * 1.0 - subAve[cnt]) * (n * 1.0 - subAve[cnt]);
		tD += total / (h * w * 1.0);
		cnt++;
	}
	//Out << "���Խ����" << log10(tD / 4.0) << endl;
	//for (int i = 0; i < 4; i++)
	//{
	//	double total = 0.0;
	//	for (int m = 0; m < h; m++)
	//		for (int n = 0; n < w; n++)
	//			total += (p[i][m][n] - subAve[i]) * (p[i][m][n] - subAve[i]);
	//	tD += total / (h * w * 1.0);
	//	Out << tD << " - ";
	//}
	//Out << (tD == 0 ? -100 : log10(tD / 4.0)) << endl;
	return tD <= 0 ? -100 : log10(tD / 4.0);
}

//double get_rsd(int** p, double& D)
//{
//	double res = 0;
//	double SD = sqrt(D);
//	vector<vector<double>> cr(1, vector<double>(2, 0));
//	for (int i = 0; i < 32; i++)
//	{
//		for (int j = 0; j < 32; j++)
//		{
//			bool flag1 = false;	// cr���Ƿ�洢�˸òв�ֵ
//			for (int k = 0; k < cr.size(); k++)
//			{
//				//Output << "����cr��" << endl;
//				//Output << cr.size() << endl;
//				if (p[i][j] == cr[k][0])	// �����ǰ�в�ֵ�Ѿ�������cr��
//				{
//					//Output << "��������" << endl;
//					cr[k][1]++;
//					flag1 = true;
//					break;
//				}
//			}
//			if (!flag1)	// ���cr��û�иòв�ֵ
//			{
//				//Output << "����������" << endl;
//				int temp[2] = { p[i][j], 1 };
//				vector<double> tempv(temp, temp + 2);
//				cr.push_back(tempv);
//			}
//		}
//	}
//
//	// ����
//	int length = cr.size();
//	for (int i = 0; i < length - 1; i++)
//	{
//		for (int j = 1; j < length - i; j++)
//		{
//			if (cr[j - 1][0] > cr[j][0])
//			{
//				int t[] = { cr[j - 1][0], cr[j - 1][1] };
//				cr[j - 1][0] = cr[j][0];
//				cr[j - 1][1] = cr[j][1];
//				cr[j][0] = t[0];
//				cr[j][1] = t[1];
//			}
//		}
//	}
//	for (int i = 0; i < length; i++)
//	{
//		cr[i][1] /= (double)1024;
//	}
//
//	double tD;
//
//	for (int i = 0; i < length; i++)
//	{
//		if (i == 0)
//		{
//			tD = normalCFD(cr[i][0] / SD);
//		}
//		else if (i == length - 1)
//		{
//			tD = 1 - normalCFD(cr[i][0] / SD);
//		}
//		else
//		{
//			double x1 = ((cr[i][0] + cr[i - 1][0]) / 2) / SD;
//			double x2 = ((cr[i][0] + cr[i + 1][0]) / 2) / SD;
//			tD = normalCFD(x2) - normalCFD(x1);
//		}
//		res += fabs(tD - cr[i][1]);
//	}
//	return res;
//}

//double normalCFD(double value)
//{
//	const double M_SQRT1_2 = sqrt(0.5);
//	return 0.5 * erfc(-value * M_SQRT1_2);
//}

bool IsN(int a, int depth)
{
	return a <= depth && a >= 0;
}

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

//vector<CxLibLinear> model13(int& flag1, int& cnt_train1_1, int& cnt_train1_0, vector<vector<double>> train_xT1, vector<vector<double>> train_xR1, vector<double> train_yT1)
//{
//	vector<CxLibLinear> res;
//	double* weight;
//	if (flag1 == 1)
//	{
//		Out << "����һ��ģ��" << endl;
//		Out << cnt_train1_1 << " --- " << cnt_train1_0 << endl;
//		weight = SetWeights(cnt_train1_1, cnt_train1_0, 1, 90, 'T', 1);
//		CxLibLinear temp1 = GetPredictor(weight[0], weight[1]);
//		temp1.train_model(train_xT1, train_yT1);
//		Out << "ѵ�����" << endl;
//		Out << weight[0] << " ^^^ " << weight[1] << endl;
//		res.push_back(temp1);
//		//weight = SetWeights(cnt_train1_1, cnt_train1_0, 1, 90, 'R', 1);
//		//CxLibLinear temp2 = GetPredictor(weight[0], weight[1]);
//		//temp1.train_model(train_xR1, train_yT1);
//		//Out << weight[0] << " ^^^ " << weight[1] << endl;
//		//res.push_back(temp2);
//	}
//	//else if (flag1 == 3)
//	//{
//	//	Out << "��������ģ��" << endl;
//	//	weight = SetWeights(cnt_train1_1, cnt_train1_0, 0, 90, 'T', 1);
//	//	CxLibLinear temp1 = GetPredictor(weight[0], weight[1]);
//	//	temp1.train_model(train_xT1, train_yT1);
//	//	res.push_back(temp1);
//	//	weight = SetWeights(cnt_train1_1, cnt_train1_0, 0, 90, 'R', 1);
//	//	CxLibLinear temp2 = GetPredictor(weight[0], weight[1]);
//	//	temp1.train_model(train_xR1, train_yT1);
//	//	res.push_back(temp2);
//	//}
//	delete[] weight;
//	Out << "��������һ�������ģ��" << endl;
//	return res;
//}

//vector<CxLibLinear> model2(int& cnt_train1_1, int& cnt_train1_0, double* weight, vector<vector<double>> train_xT1, vector<vector<double>> train_xR1, vector<double> train_yT1)
//{
//	vector<CxLibLinear> res;
//	SetWeights(cnt_train1_1, cnt_train1_0, 1, 90, 'T', 1);
//	CxLibLinear temp1 = GetPredictor(weight[0], weight[1]);
//	temp1.train_model(train_xT1, train_yT1);
//	res.push_back(temp1);
//	SetWeights(cnt_train1_1, cnt_train1_0, 1, 70, 'T', 1);
//	CxLibLinear temp2 = GetPredictor(weight[0], weight[1]);
//	temp1.train_model(train_xR1, train_yT1);
//	res.push_back(temp2);
//	return res;
//}

//vector<CxLibLinear> model3(int& cnt_train1_1, int& cnt_train1_0, double* weight, vector<vector<double>> train_xT1, vector<vector<double>> train_xR1, vector<double> train_yT1)
//{
//	vector<CxLibLinear> res;
//	SetWeights(cnt_train1_1, cnt_train1_0, 0, 90, 'T', 1);
//	CxLibLinear temp1 = GetPredictor(weight[0], weight[1]);
//	temp1.train_model(train_xT1, train_yT1);
//	res.push_back(temp1);
//	SetWeights(cnt_train1_1, cnt_train1_0, 0, 90, 'R', 1);
//	CxLibLinear temp2 = GetPredictor(weight[0], weight[1]);
//	temp1.train_model(train_xR1, train_yT1);
//	res.push_back(temp2);
//	return res;
//}

CxLibLinear CreateModel(vector<vector<double>>& x, vector<double>& y, int& cnt1, int& cnt0, int label, int demand, char TorR, int D, int qp)
{
	vector<double> weight;
	weight = SetWeights(cnt1, cnt0, label, demand, TorR, D, qp);	// ����Ȩֵ
	CxLibLinear linear = GetPredictor(weight[0], weight[1]);
	linear.train_model(x, y);
	return linear;
}

// ����ѵ����������͸����������ȷ��Ȩֵ��
vector<double> SetWeights(int cnt1, int cnt0, int label, int demand, char TorR, int D, int qp)
{
	vector<double> weight(2);
	if (D == 2 && label == 0 && TorR == 'T' && demand <= 60 && (2 * cnt0) >= (3 * cnt1))
	{
		weight[0] = 1;
		weight[1] = 10;
		return weight;
	}
	if (label)
	{
		if (demand == 90)
		{
			if (D == 0)
			{
				if (cnt1 > (cnt0))
				{
					weight[1] = 1.1 + ((cnt0 * 2.0) / cnt1);
					weight[0] = 1;
				}
			}
			if (D == 1)
			{
				if (cnt1 > (cnt0 * 4))
				{
					weight[0] = 1;
					if (TorR == 'T')
						weight[1] = 1.2;
					else if (TorR == 'R')
						weight[1] = 6;
				}
				else if (cnt1 > (cnt0 * 3))
				{
					weight[0] = 1;
					if (TorR == 'T')
						weight[1] = 1.3;
					else if (TorR == 'R')
						weight[1] = 7;
				}
				else if (cnt1 <= (cnt0 * 3) && cnt1 > cnt0)
				{
					weight[0] = 1;
					if (TorR == 'T')
						weight[1] = 4.3 - (1.0 * cnt1) / cnt0;
					else if (TorR == 'R')
						weight[1] = 13 - (2.0 * cnt1) / cnt0;
				}
				else
				{
					weight[0] = 1;
					weight[1] = 4 + (cnt0 * 2.0) / cnt1;
				}
			}
			if (D == 2)
			{
				if (cnt1 > (cnt0 * 4))
				{
					weight[0] = 1;
					if (TorR == 'T')
						weight[1] = 1.5;
					else if (TorR == 'R')
						weight[1] = 6;
				}
				else if (cnt1 > (cnt0 * 3))
				{
					weight[0] = 1;
					if (TorR == 'T')
						weight[1] = 2;
					else if (TorR == 'R')
						weight[1] = 7;
				}
				else if (cnt1 <= (cnt0 * 3) && cnt1 > cnt0)
				{
					weight[0] = 1;
					if (TorR == 'T')
						weight[1] = 2.3 - (0.3 * cnt1) / cnt0;
					else if (TorR == 'R')
						weight[1] = 13 - (2.0 * cnt1) / cnt0;
				}
				else
				{
					weight[0] = 1;
					weight[1] = 2 + (cnt0 * 1.0) / cnt1;
				}
			}
		}

		else if (demand == 70)
		{
			if (cnt1 > (cnt0 * 4))
			{
				weight[0] = 1;
				if (TorR == 'T')
					weight[1] = 0.8;
				else if (TorR == 'R')
					weight[1] = 1.5;
			}
			else if (cnt1 > cnt0 * 3)
			{
				weight[0] = 1;
				if (TorR == 'T')
					weight[1] = 1.5;
				else if (TorR == 'R')
					weight[1] = 2.2;
			}
			else if (cnt1 <= (cnt0 * 3) && cnt1 > (cnt0 * 1.0))
			{
				weight[0] = 1;
				if (TorR == 'T')
					weight[1] = 0.3 + (0.5 * cnt0) / cnt1;
				else if (TorR == 'R')
					weight[1] = 1.5 - (1.0 * cnt1) / cnt0;
			}
			else
			{
				weight[0] = 1;
				weight[1] = 1.5 + (cnt0 * 1.0) / cnt1;
			}
		}
	}

	else
	{
		if (demand == 90)
		{
			if (D == 1)
			{
				if (cnt1 > (cnt0 * 3))
				{
					if (((cnt1 * 2.0) / cnt0) > 10)
						weight[0] = 10;
					else
						weight[0] = ((cnt1 * 2.0) / cnt0);
					weight[1] = 1;
				}
				else if (cnt1 > cnt0)
				{
					if (TorR == 'T')
						weight[0] = 5 + (cnt1 * 1.0 / cnt0);
					else if (TorR == 'R')
						weight[0] = 7 + (cnt1 * 1.0 / cnt0);
					weight[1] = 1;
				}
				else
				{
					if (TorR == 'T')
					{
						if (weight[0] = 8 - (cnt0 * 1.0 / cnt1) + 10.0 / (qp - 17) <= 2)
							weight[0] = 2;
						else
							weight[0] = 8 - (cnt0 * 1.0 / cnt1) + 10.0 / (qp - 17);
					}
					else if (TorR == 'R')
					{
						if (weight[0] = 7 - (cnt0 * 1.0 / cnt1) + 10.0 / (qp - 17) + 5.0 / (qp - 17) <= 2)
							weight[0] = 2;
						else
							weight[0] = 7 - (cnt0 * 1.0 / cnt1) + 5.0 / (qp - 17);
					}
					weight[1] = 1;
				}
			}
			else if (D == 2)
			{
				if (cnt0 >= cnt1 && TorR == 'T')
				{
					weight[0] = 1.2 + (cnt1 * 0.5 / cnt0);
					weight[1] = 1;
				}
				else if (cnt0 >= cnt1 * 6 && TorR == 'R')
				{
					weight[0] = 1;
					weight[1] = 1;
				}
				else if (cnt0 >= cnt1 * 3 && cnt0 < cnt1 * 6 && TorR == 'R')
				{
					weight[0] = 1.6 - (cnt0 * 1.0) / (cnt1 * 10);
					weight[1] = 1;
				}
				else if (cnt0 >= cnt1 && cnt0 < cnt1 * 3 && TorR == 'R')
				{
					weight[0] = 1.45 - (cnt0 * 1.0) / (cnt1 * 20);
					weight[1] = 1;
				}
			}
		}
		else if (demand == 60)
		{
			if (D == 2)
			{
				if (cnt0 >= cnt1)
				{
					if (TorR == 'T')
					{
						weight[0] = 1;
						weight[1] = 1 + (cnt0 * 0.35) / cnt1;
					}
				}
				else if (cnt0 < cnt1)
				{
					if (TorR == 'T')
					{
						weight[0] = 1;
						weight[1] = 0.8 + (cnt0 * 0.25) / cnt1;
					}
				}
			}
		}
	}
	return weight;
}

// ����һ��Ԥ����
CxLibLinear GetPredictor(double weight1, double weight0)
{
	CxLibLinear linear;
	parameter param;
	param.solver_type = L2R_LR;
	param.eps = 0.001;	/* stopping criteria ֹͣ�����ı�׼�����ȣ�*/
	param.C = 2;			/*Υ��Լ����Ĵ���*/
	param.nr_weight = 2;	/*���ڸı�ĳЩ���ĳͷ�������㲻��ı��κ����͵ĳͷ���ֻҪ��nr_weight����Ϊ0��*/
	param.weight_label = Malloc(int, 2);
	param.weight_label[0] = 1;
	param.weight_label[1] = 0;
	param.weight = Malloc(double, 2);
	param.weight[0] = weight1;
	param.weight[1] = weight0;
	param.p = 0.001;			/*֧�������ع���ʧ��������*/
	param.init_sol = NULL;

	linear.init_linear_param(param);
	return linear;
}

//void Predict(vector<vector<double>>& xT, vector<vector<double>>& xR, vector<double>& y, CxLibLinear& linear, int target, char TorR, bool is_g)
//{
//	int value = -1;
//	double prob_est;
//	for (int i = 0; i < y.size(); i++)
//	{
//		if (TorR == 'T')
//			value = linear.do_predict(xT[i], prob_est);
//		else if (TorR == 'R')
//			value = linear.do_predict(xR[i], prob_est);
//		// ���Ԥ����ȷ
//
//		if (value == target)
//		{
//			if (is_g)
//			{
//				g_xT.push_back(xT[i]);
//				g_xR.push_back(xR[i]);
//				g_y.push_back(y[i]);
//				if (y[i] == 1)
//					cnt_g1++;
//				else if (y[i] == 0)
//					cnt_g0++;
//			}
//			xT.erase(xT.begin() + i);
//			xR.erase(xR.begin() + i);
//			y.erase(y.begin() + i);
//			i--;
//		}
//	}
//}