import os
import sys
import platform
from pathlib import Path

import serial
ser = serial.Serial('COM7', 9600, timeout=1)
ser.flush()

import torch

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

device = 'cpu'
weights = ROOT / 'best.pt'
dnn = False
data = ROOT / 'data/coco128.yaml'
half = False
imgsz = (640, 480)
source = '1'
vid_stride = 1
line_thickness = 3
conf_thres = 0.35
iou_thres = 0.45
agnostic_nms = False
max_det = 1000
classes = None
augment = False
update = False
hide_labels = False
hide_conf = False


centerx_camera = 640/2
centery_camera = 480/2
flame_xcentered = False
flame_ycentered = False


device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size

bs = 1  # batch_size
view_img = check_imshow(warn=True)
dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
bs = len(dataset)
# Run inference
model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
for path, im, im0s, vid_cap, s in dataset:
    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
# Inference
    with dt[1]:
        pred = model(im, augment=augment, visualize=False)
    
    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    # Process predictions
    for i, det in enumerate(pred):  # per image
        seen += 1
        p, im0, frame = path[i], im0s[i].copy(), dataset.count
        s += f'{i}: '
        s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        if len(det):
            ser.write(b"c\n")
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Show result
            xywh = []
            for *xyxy, conf, cls in reversed(det):
                if view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
            # print(len(det))
            # print(xywh)
            centerx_flame = xywh[0]+xywh[2]/2
            centery_flame = xywh[1]+xywh[3]/2 
            if flame_xcentered == False:
                if centerx_flame > centerx_camera + 20:
                    print("Fire ben phai")
                    ser.write(b"r\n")
                elif centerx_flame < centerx_camera - 20:
                    print("Fire ben trai")
                    ser.write(b"l\n")
                else:
                    print("Fire o giua")
                    ser.write(b"s\n")
                    flame_xcentered = True
            elif flame_ycentered == False:
                if centery_flame > centery_camera + 20:
                    print("Fire ben tren")
                    ser.write(b"u\n")
                elif centery_flame < centery_camera - 20:
                    print("Fire ben duoi")
                    ser.write(b"d\n")
                else:
                    print("Fire o giua")
                    ser.write(b"s\n")
                    flame_ycentered = True
            elif flame_ycentered==True and flame_xcentered==True:
                ser.write(b"p\n")
        else:
            ser.write(b"s\n")
            ser.write(b"nc\n")
            ser.write(b"np\n")
            flame_xcentered = False
            flame_ycentered = False
        # Stream results
        im0 = annotator.result()
        if view_img:
            if platform.system() == 'Linux' and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond
    #LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

# Print results
t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

if update:
    strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)