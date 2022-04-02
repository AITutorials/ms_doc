import albumentations as A
import cv2


image = cv2.imread("./input.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


transform = A.Compose(
    [
        # 运动模糊
        A.MotionBlur(),
        # 海报化
        A.Posterize(),
        # 随机雾化
        A.RandomFog(),
    ]
)


transform = A.Compose(
    [
        # 不丢失bbox下随机剪裁
        A.RandomSizedBBoxSafeCrop(),
    ]
)


transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
transformed_image = transformed["image"]
transformed_bboxes = transformed["bboxes"]
transformed_class_labels = transformed["class_labels"]


transformed = transform(image=image)["image"]

filename = "./output.png"
cv2.imwrite(filename, transformed)
