# HEVC-projects
TencCU是HM编码器中一个 非常重要的类，其中最重要的两个方法是
```cpp
Void TEncCu::compressCtu(TComDataCU* pCtu)
Void TEncCu::xCompressCU(TComDataCU*& rpcBestCU, TComDataCU*& rpcTempCU, const UInt uiDepth DEBUG_STRING_FN_DECLARE(sDebug_), PartSize eParentPartSize)
```
在这两个方法中我做了诸多修改已实现在编码过程中进行在线学习，具体包括在线生成训练集以及模型，并使用该模型进行预测。其中调用了一些函数，这些函数在FUNCs.h中
