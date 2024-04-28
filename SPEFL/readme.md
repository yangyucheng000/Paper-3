需要：pytorch,cppimport

lib目录下是cpp代码，主要是矩阵与向量运算

code目录下是python代码，各文件任务逻辑如下:
/code/CP.py:辅助计算服务器
/code/SP.py:两个秘密共享服务器
/code/Clients.py:用户
/code/getdata.py:获取数据集
/code/Models.py:网络模型定义
/code/connect.py:网络通信相关
/code/myMPC.py:安全多方计算逻辑
/code/myThread.py:多用户的多线程处理
/code/SpeflGloabal.py:全局变量定义
/code/secagg.cpp:python调用cpp代码的入口

运行:
python CP.py
python SP.py -id 0
python SP.py -id 1
python Clients.py