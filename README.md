# 点云处理过程可视化

本仓库是一个点云处理初学者为了理解点云处理过程以及PCL代码运行原理的一个demo



## 使用

### ICP

[ICP(迭代最近点)迭代过程的单步可视化程序](https://blog.csdn.net/SmileJayNew/article/details/135496381)

1. 注意修改文件的路径, PCD文件的路径不同会导致程序无法运行
2. 运行时会生成一个result.txt的文件, 里面存放的是初始生成的变换矩阵, 可以用算法运行的结果矩阵与之相对比, 判断匹配程度

```shell
git clone https://github.com/SmileJayjhx/pointcloud-processing-visualization.git
cd icp
cmake .
make
./icp_example
```

