import albumentations as A
import cv2


image = cv2.imread("./input.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
bboxes = [
    [23, 74, 295, 388],
    [377, 294, 252, 161],
    [333, 421, 49, 49],
]

class_labels = ['cat', 'dog', 'parrot']



transform = A.Compose([
    A.Perspective()
])



transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
transformed_image = transformed['image']
transformed_bboxes = transformed['bboxes']
transformed_class_labels = transformed['class_labels']



transformed = transform(image=image)["image"]

filename = "./output.png"
cv2.imwrite(filename, transformed)


