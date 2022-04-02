# 导入必备的工具包
import random
import cv2
from matplotlib import pyplot as plt
import albumentations as A


# 使用cv2读取选择的一张图片
image = cv2.imread('ob.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 找到该图片对应的标注信息，coco格式的bbox以及对应的标签
bboxes = [[5.66, 138.95, 147.09, 164.88], [366.7, 80.84, 132.8, 181.84]]
category_ids = [17, 18]

# 以及标签数值对应的实际文本
category_id_to_name = {17: 'cat', 18: 'dog'}



cv2.imwrite("test.png", image)

# 接下来我们要可视化一下这个目标检测的用例
# 我们要把这个bbox画在图片上并显示具体的标签文本


# 先定义框的颜色和文本颜色
BOX_COLOR = (0, 255, 0) # Red
TEXT_COLOR = (255, 255, 255) # White




def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """单个可视化目标框函数，参数包括，图片本身，目标框坐标，类别名字，框颜色，以及框的条纹宽度"""
    # 使用cv2.rectangle要使用极坐标，所以首先做坐标转换
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    # 先根据坐标画上目标框
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    # 设定参数来获得目标框对应的标签文本大小
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    # 再根据文本的宽高调整目标框
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)

    # 最后把文本放在目标框附近，其中设定一系列的文本参数，颜色，线条类型，字体类型等等
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    """对每一个目标框进行绘制"""
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    plt.savefig("./ob_sample.png")
    



visualize(image, bboxes, category_ids, category_id_to_name)



# 使用A构建数据增强流水线
transform = A.Compose(
    [A.RandomCropNearBBox(max_part_shift=0.3, always_apply=False, p=1.0)],
    # 因为是目标检测的数据增强，要给出bbox_params，包括重要的format
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
)

# 将图片，bbox以及类别id传入数据增强流水线
transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)


# 将得到的结果进行可视化
visualize(
    transformed['image'],
    transformed['bboxes'],
    transformed['category_ids'],
    category_id_to_name,
)

