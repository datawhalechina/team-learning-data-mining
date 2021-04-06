# 竞赛介绍
该比赛是来源于2020数字中国创新大赛-数字政府赛道。   

本赛题为智能算法赛，命题式竞赛，选题围绕“智慧海洋建设，赋能海上安全治理能力现代化”展开。提升海上安全治理能力，首要任务是“看得清”，即看得清“是什么、谁在用、做什么”。船舶避碰终端（AIS）、北斗定位终端等通信导航设备的应用，给海上交通和作业带来了极大便利，但同时存在设备信息使用不规范造成的巨大人身和财产损失，给海上安全治理带来了新的挑战。    

本赛题基于位置数据对海上目标进行智能识别和作业行为分析，要求选手通过分析渔船北斗设备位置数据，得出该船的生产作业行为，具体判断出是拖网作业、围网作业还是流刺网作业。同时，希望选手通过数据可视分析，挖掘更多海洋通信导航设备的应用价值。

# 竞赛题目
本赛题基于位置数据对海上目标进行智能识别和作业行为分析，要求选手通过分析渔船北斗设备位置数据，得出该船的生产作业行为，具体判断出是拖网作业、围网作业还是流刺网作业。初赛将提供11000条(其中7000条训练数据、2000条testA、2000条testB)渔船轨迹北斗数据。

复赛考虑以往渔船在海上作业时主要依赖AIS数据，北斗相比AIS数据，数据上报频率和数据质量均低于AIS数据，因此复赛拟加入AIS轨迹数据辅助北斗数据更好的做渔船类型识别，其中AIS数据与北斗数据的匹配需选手自行实现，具体细节复赛开赛时更新。同时，希望选手通过数据可视化与分析，挖掘更多海洋通信导航设备的应用价值。

# 竞赛数据
## 初赛
初赛提供11000条渔船北斗数据，数据包含脱敏后的渔船ID、经纬度坐标、上报时间、速度、航向信息，由于真实场景下海上环境复杂，经常出现信号丢失，设备故障等原因导致的上报坐标错误、上报数据丢失、甚至有些设备疯狂上报等。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210329225619655.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEzMzMyNw==,size_16,color_FFFFFF,t_70)

# 评估指标

以3种类别的各自F1值取平均做为评价指标，结果越大越好，具体计算公式如下：
![enter image description here](https://tianchi-public.oss-cn-hangzhou.aliyuncs.com/public/files/forum/157724301707699021577243017759.png)

其中P为某类别的准确率，R为某类别的召回率，评测程序f1函数为sklearn.metrics.f1_score，average='macro'。

# 组队学习安排

在组队学习中，我们主要是针对初赛进行复盘总结形成的baseline学习方案。   
所选用数据集为初赛的7000条训练数据、2000条testA进行操作与分析。  
其中机器学习模型的评估指标选用的是F1函数。

# 相关资料

一、比赛官网

https://tianchi.aliyun.com/competition/entrance/231768/information

二、比赛数据和地理数据分析常用工具介绍中的附件数据

链接：https://pan.baidu.com/s/1AEWhNkSzx6Ls8XmVXFOQMg

提取码：wrgg 

比赛数据在本次组队学习中只用到了hy_round1_testA_20200102与hy_round1_train_20200102文件。其中DF.csv和df_gpd_change.pkl 分别是Task1中所需要的数据。 其中DF.csv是将轨迹数据进行异常处理之后的数据，而df_gpd_change.pkl是将异常处理之后的数据进行douglas-peucker算法进行压缩之后的数据。  

其中group_df.csv是task3 模型建立后用到的数据，group_df.csv是对轨迹数据进行特征提取之后的数据，其每一列代表一个特征，每一行代表每一个船舶id。

三、比赛开源方案

1. https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586969.1002.3.163c24d1HiGiFo&postId=110644 （有源码，OTTO队伍）

2. https://tianchi.aliyun.com/forum/postDetail?postId=110932 （有源码，liu123的航空母舰）

3. https://tianchi.aliyun.com/forum/postDetail?postId=110928 （有源码，天才海神号）

4. https://tianchi.aliyun.com/forum/postDetail?postId=110710 （有源码，大白）

5. https://tianchi.aliyun.com/forum/postDetail?postId=108332 （有源码，初赛排名7,复赛排名12）

6. https://tianchi.aliyun.com/notebook-ai/detail?postId=114808 （有源码，蜗牛车，rank11）

7. https://tianchi.aliyun.com/forum/postDetail?postId=112041 （鱼佬）

8. https://tianchi.aliyun.com/forum/postDetail?postId=110943 （有源码，rank9）

