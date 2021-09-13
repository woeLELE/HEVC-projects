# HEVC-projects
如果不想看代码，可以查看《HEVC项目经历》，该文档中也有关于HEVC编码过程的介绍，可以使您更好地理解项目

TEncCU是HM编码器中一个非常重要的类，其中最重要的两个方法是
```cpp
Void TEncCu::compressCtu(TComDataCU* pCtu)
Void TEncCu::xCompressCU(TComDataCU*& rpcBestCU, TComDataCU*& rpcTempCU, const UInt uiDepth DEBUG_STRING_FN_DECLARE(sDebug_), PartSize eParentPartSize)
```
在这两个方法中我做了诸多修改以实现在编码过程中进行在线学习，前者开始于339行，原本代码很短，所以您看到的这个方法中的内容基本都是本人写的，以实现生成训练集以及模型，后者开始于720行，对其的修改集中于前面的一部分，大概结束于835行，完成了生成特征值以及进行预测的步骤。其中调用了一些函数，这些函数在FUNCs.h中
