from PyQt5.QtCore import QThread, QMutex, QTime, qDebug, QMutexLocker, pyqtSignal
from PyQt5.QtGui import QImage
from queue import Queue
import cv2
from PIL import Image
from numpy import asarray

from MatToQImage import matToQImage
from Structures import *
from Config import *
# from track import Tack_Object #kien

#kien area
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort



class Tack_Object:
    def __init__(self,yolo_model, deep_sort_model, source, output, imgsz, conf_thres, iou_thres, fourcc,
               device, show_vid, save_vid, save_txt, classes, agnostic_nms, augment, evaluate, config_deepsort,
               half, visualize, max_det, dnn, project, name, exist_ok):
        self.yolo_model = yolo_model
        self.deep_sort_model = deep_sort_model
        self.output = output
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.show_vid = show_vid
        self.save_vid = save_vid
        self.save_txt = save_txt
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.evaluate = evaluate
        self.config_deepsort = config_deepsort
        self.half = half
        self.visualize = visualize
        self.max_det = max_det
        self.dnn = dnn
        self.project = project
        self.name = name
        self.exist_ok = exist_ok


    # def detect(opt):
    #     out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok= \
    #         opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
    #         opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    # def detect(self, out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok):
    def detect(self,yolo_model, deep_sort_model, source, output, imgsz, conf_thres, iou_thres, fourcc,
               device, show_vid, save_vid, save_txt, classes, agnostic_nms, augment, evaluate, config_deepsort,
               half, visualize, max_det, dnn, project, name, exist_ok):
        out = output
        webcam = source == '0' or source.startswith(
            'rtsp') or source.startswith('http') or source.endswith('.txt')

        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(config_deepsort)
        deepsort = DeepSort(deep_sort_model,
                            max_dist=cfg.DEEPSORT.MAX_DIST,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        # Initialize
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA

        # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
        # its own .txt file. Hence, in that case, the output folder is not restored
        if not evaluate:
            if os.path.exists(out):
                pass
                shutil.rmtree(out)  # delete output folder
            os.makedirs(out)  # make new output folder

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(yolo_model, device=device, dnn=dnn)
        stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Half
        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()

        # Set Dataloader
        vid_path, vid_writer = None, None
        # Check if environment supports image displays
        if show_vid:
            show_vid = check_imshow()
            print("ák",source)

        # Dataloader
        if webcam:
            show_vid = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs
        # print("dt", dataset)
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        # extract what is in between the last '/' and last '.'
        txt_file_name = source.split('/')[-1].split('.')[0]
        txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
            t1 = time_sync()
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            t2 = time_sync()
            dt[0] += t2 - t1
            self.currentFrame = img
            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, _ = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                s += '%gx%g ' % img.shape[2:]  # print string

                annotator = Annotator(im0, line_width=2, pil=not ascii)
                # print("a1",annotator)
                # print("i1", im0)
                w, h = im0.shape[1],im0.shape[0]
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    # pass detections to deepsort
                    t4 = time_sync()
                    outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                    t5 = time_sync()
                    dt[3] += t5 - t4

                    # draw boxes for visualization
                    if len(outputs) > 0:
                        for j, (output, conf) in enumerate(zip(outputs, confs)):

                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]
                            #count
                            count_obj(bboxes,w,h,id)
                            c = int(cls)  # integer class
                            label = f'{id} {names[c]} {conf:.2f}'
                            annotator.box_label(bboxes, label, color=colors(c, True))

                            if save_txt:
                                # to MOT format
                                bbox_left = output[0]
                                bbox_top = output[1]
                                bbox_w = output[2] - output[0]
                                bbox_h = output[3] - output[1]
                                # Write MOT compliant results to file
                                with open(txt_path, 'a') as f:
                                    f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                                   bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

                    LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

                else:
                    deepsort.increment_ages()
                    LOGGER.info('No detections')

                # Stream results
                im0 = annotator.result()
                if show_vid:
                    global count
                    color=(0,255,0)
                    start_point = (0, h-350)
                    end_point = (w, h-350)
                    cv2.line(im0, start_point, end_point, color, thickness=2)
                    thickness = 3
                    org = (150, 150)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 3
                    cv2.putText(im0, str(count), org, font,
                       fontScale, color, thickness, cv2.LINE_AA)
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_vid:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]

                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
            per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_vid:
            print('Results saved to %s' % save_path)
            if platform == 'darwin':  # MacOS
                os.system('open ' + save_path)

def count_obj(box,w,h,id):
    global count,data
    center_coordinates = (int(box[0]+(box[2]-box[0])/2) , int(box[1]+(box[3]-box[1])/2))
    if int(box[1]+(box[3]-box[1])/2) > (h -350):
        if  id not in data:
            count += 1
            data.append(id)
            print('count',count)
#kien area


class ProcessingThread(QThread):
    newFrame = pyqtSignal(QImage)
    updateStatisticsInGUI = pyqtSignal(ThreadStatisticsData)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def __init__(self, sharedImageBuffer, deviceUrl, cameraId, parent=None):
        super(QThread, self).__init__(parent)
        self.sharedImageBuffer = sharedImageBuffer
        self.cameraId = cameraId
        # Save Device Url
        self.deviceUrl = deviceUrl
        # Initialize members
        self.doStopMutex = QMutex()
        self.processingMutex = QMutex()
        self.t = QTime()
        self.processingTime = 0
        self.doStop = False
        self.enableFrameProcessing = False
        self.sampleNumber = 0
        self.fpsSum = 0.0
        self.fps = Queue()
        self.currentROI = QRect()
        self.imgProcFlags = ImageProcessingFlags()
        self.imgProcSettings = ImageProcessingSettings()
        self.statsData = ThreadStatisticsData()
        self.frame = None
        self.currentFrame = None

    def run(self):
        while True:
            ##############################
            # Stop thread if doStop=True #
            ##############################
            self.doStopMutex.lock()
            if self.doStop:
                self.doStop = False
                self.doStopMutex.unlock()
                break
            self.doStopMutex.unlock()
            ################################
            ################################

            # Save processing time
            self.processingTime = self.t.elapsed()
            # Start timer (used to calculate processing rate)
            self.t.start()

            with QMutexLocker(self.processingMutex):
                # Get frame from queue, store in currentFrame, set ROI
                # self.currentFrame = Mat(self.sharedImageBuffer.getByDeviceUrl(self.deviceUrl).get().clone(),
                #                         self.currentROI)
                self.currentFrame = self.sharedImageBuffer.getByDeviceUrl(self.deviceUrl).get()[
                                    self.currentROI.y():(self.currentROI.y() + self.currentROI.height()),
                                    self.currentROI.x():(self.currentROI.x() + self.currentROI.width())].copy()

                # Example of how to grab a frame from another stream (where Device Url=1)
                # Note: This requires stream synchronization to be ENABLED (in the Options menu of MainWindow)
                #       and frame processing for the stream you are grabbing FROM to be DISABLED.
                # if sharedImageBuffer.containsImageBufferForDeviceUrl(1):
                #     # Grab frame from another stream (connected to camera with Device Url=1)
                #     Mat frameFromAnotherStream = Mat(sharedImageBuffer.getByDeviceUrl(1).getFrame(), currentROI)
                #     # Linear blend images together using OpenCV and save the result to currentFrame. Note: beta=1-alpha
                #     addWeighted(frameFromAnotherStream, 0.5, currentFrame, 0.5, 0.0, currentFrame)

                ##################################
                # PERFORM IMAGE PROCESSING BELOW #
                ##################################

                # Grayscale conversion (in-place operation)
                if self.imgProcFlags.grayscaleOn and (
                        self.currentFrame.shape[2] == 3 or self.currentFrame.shape[2] == 4):
                    self.currentFrame = cv2.cvtColor(self.currentFrame, cv2.COLOR_BGR2GRAY)

                # Smooth (in-place operations)
                if self.imgProcFlags.smoothOn:
                    if self.imgProcSettings.smoothType == 0:
                        # BLUR
                        self.currentFrame = cv2.blur(self.currentFrame,
                                                     (self.imgProcSettings.smoothParam1,
                                                      self.imgProcSettings.smoothParam2))
                    elif self.imgProcSettings.smoothType == 1:
                        # GAUSSIAN
                        self.currentFrame = cv2.GaussianBlur(self.currentFrame,
                                                             (self.imgProcSettings.smoothParam1,
                                                              self.imgProcSettings.smoothParam2),
                                                             sigmaX=self.imgProcSettings.smoothParam3,
                                                             sigmaY=self.imgProcSettings.smoothParam4)
                    elif self.imgProcSettings.smoothType == 2:
                        # MEDIAN
                        self.currentFrame = cv2.medianBlur(self.currentFrame, self.imgProcSettings.smoothParam1)

                # Dilate
                if self.imgProcFlags.dilateOn:
                    self.currentFrame = cv2.dilate(self.currentFrame, self.kernel,
                                                   iterations=self.imgProcSettings.dilateNumberOfIterations)
                # Erode
                if self.imgProcFlags.erodeOn:
                    self.currentFrame = cv2.erode(self.currentFrame, self.kernel,
                                                  iterations=self.imgProcSettings.erodeUrlOfIterations)
                # Flip
                if self.imgProcFlags.flipOn:
                    self.currentFrame = cv2.flip(self.currentFrame, self.imgProcSettings.flipCode)
                    a=1

                #YoloV5 Track
                if self.imgProcFlags.yolov5:
                    a=1
                    image1 = Image.open('videos/Car.jpg')
                    data1 = asarray(image1)
                    image2 = Image.open('videos/dj.jpg')
                    data2 = asarray(image2)

                    # print('Image saved as:', self.deviceUrlEdit.text())
                    # PIL_image = Image.fromarray(self.currentFrame)
                    # PIL_image.save('Car.jpg')



                    FILE = Path(__file__).resolve()
                    ROOT = FILE.parents[0]  # yolov5 deepsort root directory
                    if str(ROOT) not in sys.path:
                        sys.path.append(str(ROOT))  # add ROOT to PATH
                    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
                    count = 0
                    data = []
                    for chu in 'quantrimang':
                        print('Chữ cái hiện tại:', chu)

                    # Lặp từ trong chuỗi
                    chuoi = ['bố', 'mẹ', 'em']
                    for tu in chuoi:
                        print('Anh yêu', tu)
                    def a():
                        print("a")
                        self.currentFrame = data1
                    def b():
                        print("b")
                        self.currentFrame = data2

                    a()
                    b()
                    #
                    # a = Tack_Object(yolo_model='yolov5s.pt', deep_sort_model='osnet_x0_25', source='videos/Traffic.mp4',
                    #                 output='inference/output', imgsz=[480, 480], conf_thres=0.5, iou_thres=0.5,
                    #                 fourcc='mp4v', device='',
                    #                 show_vid=False, save_vid=False, save_txt=False, classes=2, agnostic_nms=False,
                    #                 augment=False,
                    #                 evaluate=False,
                    #                 config_deepsort='deep_sort/configs/deep_sort.yaml', half=False, visualize=False,
                    #                 max_det=1000,
                    #                 dnn=False,
                    #                 project="WindowsPath('runs/track')", name='exp', exist_ok=False)
                    # a.detect(yolo_model='yolov5s.pt', deep_sort_model='osnet_x0_25', source='videos/Traffic.mp4',
                    #          output='inference/output', imgsz=[480, 480], conf_thres=0.5, iou_thres=0.5, fourcc='mp4v',
                    #          device='',
                    #          show_vid=False, save_vid=False, save_txt=False, classes=2, agnostic_nms=False,
                    #          augment=False, evaluate=False,
                    #          config_deepsort='deep_sort/configs/deep_sort.yaml', half=False, visualize=False,
                    #          max_det=1000, dnn=False,
                    #          project="WindowsPath('runs/track')", name='exp', exist_ok=False)
                # Canny edge detection
                if self.imgProcFlags.cannyOn:
                    self.currentFrame = cv2.Canny(self.currentFrame,
                                                  threshold1=self.imgProcSettings.cannyThreshold1,
                                                  threshold2=self.imgProcSettings.cannyThreshold2,
                                                  apertureSize=self.imgProcSettings.cannyApertureSize,
                                                  L2gradient=self.imgProcSettings.cannyL2gradient)

                ##################################
                # PERFORM IMAGE PROCESSING ABOVE #
                ##################################

                # Convert Mat to QImage
                self.frame = matToQImage(self.currentFrame)

                # Inform GUI thread of new frame (QImage)
                self.newFrame.emit(self.frame)

            # Update statistics
            self.updateFPS(self.processingTime)
            self.statsData.nFramesProcessed += 1
            # Inform GUI of updated statistics
            self.updateStatisticsInGUI.emit(self.statsData)

        qDebug("Stopping processing thread...")

    def doShowImage(self, val):
        with QMutexLocker(self.processingMutex):
            self.doShow = val

    def updateFPS(self, timeElapsed):
        # Add instantaneous FPS value to queue
        if timeElapsed > 0:
            self.fps.put(1000 / timeElapsed)
            # Increment sample number
            self.sampleNumber += 1

        # Maximum size of queue is DEFAULT_PROCESSING_FPS_STAT_QUEUE_LENGTH
        if self.fps.qsize() > PROCESSING_FPS_STAT_QUEUE_LENGTH:
            self.fps.get()

        # Update FPS value every DEFAULT_PROCESSING_FPS_STAT_QUEUE_LENGTH samples
        if self.fps.qsize() == PROCESSING_FPS_STAT_QUEUE_LENGTH and self.sampleNumber == PROCESSING_FPS_STAT_QUEUE_LENGTH:
            # Empty queue and store sum
            while not self.fps.empty():
                self.fpsSum += self.fps.get()
            # Calculate average FPS
            self.statsData.averageFPS = self.fpsSum / PROCESSING_FPS_STAT_QUEUE_LENGTH
            # Reset sum
            self.fpsSum = 0.0
            # Reset sample number
            self.sampleNumber = 0

    def stop(self):
        with QMutexLocker(self.doStopMutex):
            self.doStop = True

    def updateBoxesBufferMax(self, boxesBufferMax):
        with QMutexLocker(self.processingMutex):
            self.boxesBufferMax = boxesBufferMax

    def updateImageProcessingFlags(self, imgProcFlags):
        with QMutexLocker(self.processingMutex):
            self.imgProcFlags.grayscaleOn = imgProcFlags.grayscaleOn
            self.imgProcFlags.smoothOn = imgProcFlags.smoothOn
            self.imgProcFlags.dilateOn = imgProcFlags.dilateOn
            self.imgProcFlags.erodeOn = imgProcFlags.erodeOn
            self.imgProcFlags.flipOn = imgProcFlags.flipOn
            self.imgProcFlags.cannyOn = imgProcFlags.cannyOn
            self.imgProcFlags.yolov5 = imgProcFlags.yolov5

    def updateImageProcessingSettings(self, imgProcSettings):
        with QMutexLocker(self.processingMutex):
            self.imgProcSettings.smoothType = imgProcSettings.smoothType
            self.imgProcSettings.smoothParam1 = imgProcSettings.smoothParam1
            self.imgProcSettings.smoothParam2 = imgProcSettings.smoothParam2
            self.imgProcSettings.smoothParam3 = imgProcSettings.smoothParam3
            self.imgProcSettings.smoothParam4 = imgProcSettings.smoothParam4
            self.imgProcSettings.dilateNumberOfIterations = imgProcSettings.dilateNumberOfIterations
            self.imgProcSettings.erodeUrlOfIterations = imgProcSettings.erodeUrlOfIterations
            self.imgProcSettings.flipCode = imgProcSettings.flipCode
            self.imgProcSettings.cannyThreshold1 = imgProcSettings.cannyThreshold1
            self.imgProcSettings.cannyThreshold2 = imgProcSettings.cannyThreshold2
            self.imgProcSettings.cannyApertureSize = imgProcSettings.cannyApertureSize
            self.imgProcSettings.cannyL2gradient = imgProcSettings.cannyL2gradient

    def setROI(self, roi):
        with QMutexLocker(self.processingMutex):
            self.currentROI.setX(roi.x())
            self.currentROI.setY(roi.y())
            self.currentROI.setWidth(roi.width())
            self.currentROI.setHeight(roi.height())

    def getCurrentROI(self):
        return QRect(self.currentROI.x(), self.currentROI.y(), self.currentROI.width(), self.currentROI.height())

