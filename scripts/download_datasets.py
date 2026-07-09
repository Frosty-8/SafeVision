from pycocotools.coco import COCO

coco = COCO("data/coco/annotations/instances_val2017.json")

print(len(coco.imgs))
print(len(coco.anns))
print(len(coco.cats))
