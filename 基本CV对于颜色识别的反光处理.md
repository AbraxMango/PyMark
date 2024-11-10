# 基本CV对于颜色识别的反光处理

> 由于在类似于智能车比赛的线下赛中，常常会受到强光干扰，试得视觉处理不清楚，导致小车失去了“眼睛”
>
> 而通过简单的处理可以有效减少反光造成的影响。

- 创建掩码提取白色

由于白色常常是场地的基础色，即不需要进行识别的部分，而光照反射导致的后果就是一片难以识别的白色区域。

我们通过设定的阈值进行颜色判断后，常常是通过边缘寻找而确定颜色区域的中心而确定目标。这个时候我们就

可以通过设置掩码将白色部分去除而不影响原本颜色的识别

代码 -> 

```py
# 转换到HSV色彩空间（更容易提取颜色范围）
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 设置白色的HSV范围
# 白色的色调范围宽泛，设置较高的阈值来获取白色区域
lower_white = np.array([0, 0, 120])  # H: 0, S: 0, V: 200 (亮度值)
upper_white = np.array([180, 50, 255])  # H: 180, S: 50, V: 255 (亮度范围)

# 创建掩码，将白色区域提取出来
mask = cv2.inRange(hsv, lower_white, upper_white)

# 将白色区域替换为黑色或其他颜色
image_no_white = cv2.bitwise_and(image, image, mask=~mask)  # 使用反掩码去除白色区域
cv2.imshow('Image without white parts', image_no_white)  # 显示经过处理后的图像 
```

<img src="C:\Users\肖逸轩\AppData\Roaming\Typora\typora-user-images\image-20241110135424948.png" alt="image-20241110135424948" style="zoom:50%;" />

图1：未进行掩码操作的效果

<img src="C:\Users\肖逸轩\AppData\Roaming\Typora\typora-user-images\image-20241110140103080.png" alt="image-20241110140103080" style="zoom:50%;" />

图2：进行掩码操作后的效果

完整函数代码 ->

```py
def find_color_center(color, image, lower_color, upper_color, min_area):
    """
    找到指定颜色范围内最大像素块的中心坐标，如果面积大于最小面积。
    :param color: 颜色
    :param image: 输入图像
    :param lower_color: 颜色范围下限 (LAB)
    :param upper_color: 颜色范围上限 (LAB)
    :param min_area: 最小面积阈值
    :return: 最大像素块的中心坐标 (cX, cY)，如果找到；否则返回 None
    """
    # 转换到HSV色彩空间（更容易提取颜色范围）
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 设置白色的HSV范围
    # 白色的色调范围宽泛，设置较高的阈值来获取白色区域
    lower_white = np.array([0, 0, 120])  # H: 0, S: 0, V: 200 (亮度值)
    upper_white = np.array([180, 50, 255])  # H: 180, S: 50, V: 255 (亮度范围)

    # 创建掩码，将白色区域提取出来
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 将白色区域替换为黑色或其他颜色
    image_no_white = cv2.bitwise_and(image, image, mask=~mask)  # 使用反掩码去除白色区域
    cv2.imshow('Image without white parts', image_no_white)

    dem = cv2.cvtColor(image_no_white, cv2.COLOR_BGR2LAB)
    # 创建颜色范围内的掩码
    mask = cv2.inRange(dem, lower_color, upper_color)

    # 开闭操作
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)  # 开操作去除小的亮点
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element)  # 闭操作填补小的空洞

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

```

