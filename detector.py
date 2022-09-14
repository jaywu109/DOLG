import sys
import torch
from yolov4_scaled.models.experimental import attempt_load
from yolov4_scaled.utils.general import check_img_size, non_max_suppression, scale_coords

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.append('yolov4_scaled')


class _Detector:
    # Meta Class
    def __init__(self):
        # load model paramter here
        self.model = None
        self.model.eval()

    def predict(self, img_tensor):
        """Return coords with (num_samples, 4) correspond to x0, y0, x1, y1
        """
        raise NotImplementedError('Child class must implement this method')


class YoloV4ScaledDetector(_Detector):

    def __init__(self, model_weights, img_size=640, conf_thres=0.4, iou_thres=0.5, agnostic_nms=False, save_img=False):
        model = attempt_load(model_weights, map_location=DEVICE)
        self.model = model.to(DEVICE).eval()
        self.img_size = check_img_size(img_size, s=model.stride.max())
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.agnostic_nms = agnostic_nms
        self.save_img = save_img

    def predict(self, img, im0):
        img = torch.from_numpy(img).to(DEVICE).float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():
            pred = self.model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=self.agnostic_nms)

        # Process detections
        result = []
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in det:
                    xyxy = torch.tensor(xyxy)
                    result.append((xyxy, conf, cls))
        return result
