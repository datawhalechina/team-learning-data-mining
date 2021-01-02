# 任务5：作者信息关联

## 5.1 任务说明

- 学习主题：作者关联（数据建模任务），对论文作者关系进行建模，统计最常出现的作者关系；
- 学习内容：构建作者关系图，挖掘作者关系
- 学习成果：论文作者知识图谱、图关系挖掘

## 5.2 数据处理步骤

将作者列表进行处理，并完成统计。具体步骤如下：

- 将论文第一作者与其他作者（论文非第一作者）构建图；
- 使用图算法统计图中作者与其他作者的联系；

## 5.3 社交网络分析

图是复杂网络研究中的一个重要概念。Graph是用**点**和**线**来刻画离散事物集合中的每对事物间以某种方式相联系的数学模型。Graph在现实世界中随处可见，如交通运输图、旅游图、流程图等。利用图可以描述现实生活中的许多事物，如用点可以表示交叉口，点之间的连线表示路径，这样就可以轻而易举的描绘出一个交通运输网络。

### 5.3.1 图类型

- 无向图，忽略了两节点间边的方向。

- 指有向图，考虑了边的有向性。

- 多重无向图，即两个结点之间的边数多于一条，又允许顶点通过同一条边和自己关联。

### 5.3.2 图统计指标

- 度：是指和该节点相关联的边的条数，又称关联度。对于有向图，节点的入度 是指进入该节点的边的条数；节点的出度是指从该节点出发的边的条数；

- 迪杰斯特拉路径：.从一个源点到其它各点的最短路径，可使用迪杰斯特拉算法来求最短路径；

- 连通图：在一个无向图 G 中，若从顶点i到顶点j有路径相连，则称i和j是连通的。如果 G 是有向图，那么连接i和j的路径中所有的边都必须同向。如果图中任意两点都是连通的，那么图被称作连通图。如果此图是有向图，则称为强连通图。

对于其他图算法，可以在networkx和igraph两个库中找到。

## 5.4 具体代码以及讲解

首先读取我们想要的数据：

```python
data  = [] #初始化
#使用with语句优势：1.自动关闭文件句柄；2.自动显示（处理）文件读取数据异常
with open("arxiv-metadata-oai-snapshot.json", 'r') as f: 
    for idx, line in enumerate(f): 
        d = json.loads(line)
        d = {'authors_parsed': d['authors_parsed']}
        data.append(d)
        
data = pd.DataFrame(data) #将list变为dataframe格式，方便使用pandas进行分析
```

创建作者链接的无向图：

```python
import networkx as nx 
# 创建无向图
G = nx.Graph()

# 只用五篇论文进行构建
for row in data.iloc[:5].itertuples():
    authors = row[1]
    authors = [' '.join(x[:-1]) for x in authors]
    
    # 第一个作者 与 其他作者链接
    for author in authors[1:]:
        G.add_edge(authors[0],author) #　添加节点２，３并链接２３节点
```

将作者关系图进行绘制：

```python
nx.draw(G, with_labels=True)
```

<img src="img/task5_image1.png" alt="task5_image1" style="zoom:50%;"/>

得到作者之间的距离：

```python
try:
    print(nx.dijkstra_path(G, 'Balázs C.', 'Ziambaras Eleni'))
except:
    print('No path')
```

如果我们500片论文构建图，则可以得到更加完整作者关系，并选择最大联通子图进行绘制，折线图为子图节点度值。

```python
# 计算论文关系中有多少个联通子图
print(len(nx.communicability(G)))

plt.loglog(degree_sequence, "b-", marker="o")
plt.title("Degree rank plot")
plt.ylabel("degree")
plt.xlabel("rank")

# draw graph in inset
plt.axes([0.45, 0.45, 0.45, 0.45])
Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])

pos = nx.spring_layout(Gcc)
plt.axis("off")
nx.draw_networkx_nodes(Gcc, pos, node_size=20)
nx.draw_networkx_edges(Gcc, pos, alpha=0.4)
plt.show()
```

<img src="img/task5_image2.png" alt="task5_image2" style="zoom:50%;"/>
