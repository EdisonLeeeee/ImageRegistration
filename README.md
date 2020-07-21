# Image Registration
图像配准是将不同时间、不同传感器、不同条件下获得的两幅或者多幅图像进行匹配叠加的过程，是多源数据联合处理的基础，被广泛应用于遥感图像分析、导航制导、智能视觉等领域。


+ 实验数据包含两个文件夹：RealImg和RefImg，其中RealImg里面存储的是无人机获取的实时图像，RefImg文件夹里面存储的是卫星获得的基准图像，实时图和基准图是成对出现（如RealImg_0.bmp和RefImg_0.bmp是一对），利用图像处理课程中所学知识，完成所有实时图与相应基准图的配准；
+ 按照图像后缀数字顺序，依次输出实时图在基准图中两个方向的配准偏移量（∆x,∆y）,x表示横向，y表示纵向，两幅图像都以（1，1）坐标为起始参考坐标。输出结果存储在txt文件里面，每一行存储一组图像的配准偏移量，∆x,∆y中间用空格隔开,见示例MatchResult.txt文件；
+ Image I: Reference image (参考图)
+ Image J: Real Image （实时图）

![Reg](./RegImg/RegImg_0.png)

# Requirements
+ Python3
+ Opencv-python
+ Opencv-contrib-python
+ scikit_image
+ scipy
+ numpy
+ matplotlib
+ Pillow

# Usage
```python
python reg.py
```
For more details, you can see `demo.ipynb`.
