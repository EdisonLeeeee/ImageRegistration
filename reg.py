import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage.io import imsave

from PIL import Image
from PIL import ImageEnhance
from skimage.exposure import match_histograms


def ImageRegistration(I, J, dx, dy):
    """
    图像配准，给定偏移量，将两张图像合成
    
    Arguments:
    -----------
    I: 输入的基准图像
    J：实时图像（待配准）
    dx：实时图像在基准图像 （高度）Y 轴上的偏移
    dy：实时图像在基准图像 （宽度）X 轴上的偏移
    
    Returns:
    -----------
    配准后的图像
    
    """
    h1, w1 = I.shape
    h2, w2 = J.shape
    img = I.copy()
    img[dx: min(dx+h2, h1), dy:min(dy+w2, w1)] = J[:min(h2, h1-dx), :min(w2, w1-dy)]
    return img

def plot(I, J, dx, dy, file_name):
    """
    显示四张图像并保存
    1. 参考图像
    2. 实时图像
    3. 参考图像中与实时图像配准部分的图像
    4. 配准后的图像
    
    Arguments:
    -----------
    I: 输入的基准图像
    J：实时图像（待配准）
    dx：实时图像在基准图像 （高度）Y 轴上的偏移
    dy：实时图像在基准图像 （宽度）X 轴上的偏移
    file_name：保存的图片名称
    
    """       
    h1, w1 = I.shape
    h2, w2 = J.shape
    img = ImageRegistration(I, J, dx, dy)
    fig = plt.figure(figsize=(30,30))
    fig.add_subplot(2,2,1)
    plt.imshow(I, cmap='gray')
    plt.title("Reference Image", fontsize=30)
    fig.add_subplot(2,2,2)
    plt.imshow(img, cmap='gray') 
    plt.title("Registered Image", fontsize=30)    
    fig.add_subplot(2,2,3)
    plt.imshow(J, cmap='gray') 
    plt.title("Real Image", fontsize=30)
    fig.add_subplot(2,2,4)
    plt.imshow(I[dx:dx+h2, dy:dy+w2], cmap='gray')
    plt.title("Selected Part of Reference Image", fontsize=30)
    plt.tight_layout()
    plt.savefig(f'{file_name}.png')
    plt.close('all') #关闭所有 figure windows


def detect(I, J):
    
    # 图像亮度增强
    J = ImageEnhance.Brightness(Image.fromarray(J)).enhance(1.8)
    J = np.asarray(J)

    # SIFT 检测算法
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_I, descriptors_I= sift.detectAndCompute(I, None)
    keypoints_J, descriptors_J = sift.detectAndCompute(J, None)

    # 暴力搜索匹配
    bf = cv2.BFMatcher(crossCheck=False)
    # 最近邻匹配
    matches = bf.knnMatch(descriptors_I, descriptors_J, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.825*n.distance:
            good.append([m])

    # 获取匹配良好的特征点坐标
    points1 = np.zeros((len(good), 2), dtype=np.float32)
    points2 = np.zeros((len(good), 2), dtype=np.float32)

    for i, match in enumerate(good):
        points1[i, :] = keypoints_I[match[0].queryIdx].pt
        points2[i, :] = keypoints_J[match[0].trainIdx].pt
        
    # 计算偏移量
    dx, dy = (points1-points2).mean(0).astype('uint8').tolist() 
    return dx, dy
        
        
if __name__ == '__main__':
    
    # 创建图片存储路径
    if not os.path.exists("RegImg"):
        os.makedirs("RegImg")

    results = []
    for idx in range(97):
        # 读取图片 数据
        I = io.imread(f'RefImg/RefImg_{idx}.bmp').astype('uint8')
        J = io.imread(f'RealImg/RealImg_{idx}.bmp').astype('uint8')

        # 图像配准，计算偏移量
        dx, dy = detect(I, J)
        results.append(f'{dy} {dx}\n')
        plot(I, J, dx, dy, f'RegImg/RegImg_{idx}')


    # 写入结果
    with open('MatchResult.txt', 'w') as f:
        f.writelines(results)

    print(''.join(results))
