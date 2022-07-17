# from track import Tack_Object
from A2 import Tack_Object
from PIL import Image
from numpy import asarray
# load the image
image = Image.open('videos/Car.jpg')
# convert image to numpy array
data = asarray(image)

PIL_image = Image.fromarray(data)
PIL_image.save('Car.jpg')
print("DT",data)
print("DTq",PIL_image)
# 'videos/Traffic.mp4'
yolo_model='yolov5s.pt'
deep_sort_model='osnet_x0_25'
source="videos/Traffic.mp4"
output='inference/output'
imgsz=[480, 480]
conf_thres=0.5
iou_thres=0.5
fourcc='mp4v'
device=''
show_vid=False
save_vid=False
save_txt=False
classes=2
agnostic_nms=False
augment=False
evaluate=False
config_deepsort='deep_sort/configs/deep_sort.yaml'
half=False
visualize=False
max_det=1000
dnn=False
project="WindowsPath('runs/track')"
name='exp'
exist_ok=False

Tack_Object(yolo_model, deep_sort_model, source, output, imgsz, conf_thres, iou_thres, fourcc,
               device, show_vid, save_vid, save_txt, classes, agnostic_nms, augment, evaluate, config_deepsort,
               half, visualize, max_det, dnn, project, name, exist_ok)



# Tack_Object(yolo_model='yolov5s.pt', deep_sort_model='osnet_x0_25', source="videos/Traffic.mp4",
#                 output='inference/output', imgsz=[480, 480], conf_thres=0.5, iou_thres=0.5, fourcc='mp4v', device='',
#                 show_vid=False, save_vid=False, save_txt=False, classes=2, agnostic_nms=False, augment=False,
#                 evaluate=False,
#                 config_deepsort='deep_sort/configs/deep_sort.yaml', half=False, visualize=False, max_det=1000,
#                 dnn=False,
#                 project="WindowsPath('runs/track')", name='exp', exist_ok=False)

# a.detect(yolo_model='yolov5s.pt', deep_sort_model='osnet_x0_25', source="videos/Traffic.mp4",
#          output='inference/output', imgsz=[480, 480], conf_thres=0.5, iou_thres=0.5, fourcc='mp4v', device='',
#          show_vid=False, save_vid=False, save_txt=False, classes=2, agnostic_nms=False, augment=False, evaluate=False,
#          config_deepsort='deep_sort/configs/deep_sort.yaml', half=False, visualize=False, max_det=1000, dnn=False,
#          project="WindowsPath('runs/track')", name='exp', exist_ok=False)