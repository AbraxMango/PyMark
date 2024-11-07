# 工创赛-OPENCV识别小球

> 基于cv2的cv.cvtColor方法来基于颜色阈值识别小球

```python
import cv2
import numpy as np

def find_color_center(color, image, lower_color, upper_color, min_area):
    """
    找到指定颜色范围内最大像素块的中心坐标，如果面积大于最小面积。
    :param color: 小球颜色
    :param image: 输入图像
    :param lower_color: 颜色范围下限 (HSV)
    :param upper_color: 颜色范围上限 (HSV)
    :param min_area: 最小面积阈值
    :return: 最大像素块的中心坐标 (cX, cY)，如果找到；否则返回 None
    """
    if color == "red":
        # 将图像转换到LAB色彩空间
        dem = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    else:
		    
        dem = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 创建颜色范围内的掩码
    mask = cv2.inRange(dem, lower_color, upper_color)

    # 开闭操作
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element)

    # 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化最大面积和对应的中心点
    max_area = 0
    max_area_center = None

    # 遍历轮廓并计算中心坐标
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area and area > min_area:  # 检查面积是否大于最小面积
            max_area = area
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                max_area_center = (cX, cY)

    return max_area_center, mask

def main():
    # 读取图像并调整大小
    image = cv2.imread('data/p2.png')
    resized_image = cv2.resize(image, (640, 480))

    # 定义颜色范围 (HSV) 和最小面积阈值
    color_ranges = {
        "red": (np.array([0, 148, 0]), np.array([255, 194, 255]), 500),
        "yellow": (np.array([20, 90, 116]), np.array([40, 255, 255]), 500),
        "blue": (np.array([100, 150, 50]), np.array([140, 255, 255]), 500),
        "black": (np.array([0, 0, 0]), np.array([180, 255, 50]), 500)
    }

    # 按顺序检测颜色
    for color, (lower, upper, min_area) in color_ranges.items():
        center, mask = find_color_center(color, resized_image, lower, upper, min_area)
        if center:
            # 绘制绿色十字键头
            cv2.line(resized_image, (center[0] - 10, center[1]), (center[0] + 10, center[1]), (0, 255, 0), 2)
            cv2.line(resized_image, (center[0], center[1] - 10), (center[0], center[1] + 10), (0, 255, 0), 2)
            print(f"{color.capitalize()} pos:{center}")
            cv2.imshow(f'{color.capitalize()} Mask', mask)
            # break  # 检测到颜色后退出
    else:
        print("None")

    # 显示结果图像
    cv2.imshow('Result', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

代码解析：

1. 首先读入图像并调整尺寸
2. 设置颜色阈值，其中红色的阈值为Lab算子的阈值，其他颜色为Hsv算子阈值
3. 按顺序检测颜色
4. 如果为红色，则使用Lab算子(不知道为什么Hsv算子识别红色的时候总是会缺一小块)，其他颜色则用Hsv算子
5. 根据颜色范围创建掩码之后进行开闭操作对图像进行简单的处理
6. 绘出轮廓，计算中心点
7. 结束