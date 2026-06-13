#!/usr/bin/env python3
"""yolo_delivery_node.py — YOLOv11 on the delivery cameras, GPU via TensorRT.

The installed torch (2.4.1) is a CPU-only build, so ultralytics YOLO(best.pt)
ran inference ON CPU -> badly laggy. This node instead runs the prebuilt
TensorRT FP16 engine (best.engine, imgsz from the engine) with pycuda -> real
GPU inference (the same path the standalone realsense_yolo_trt.py uses).

Build the engine once (matches best.pt) if it is missing/stale:
  yolo export model=best.pt format=engine half=True imgsz=320   # or trtexec

Subscribes ~image_topic, publishes:
  <out_ns>/detections  std_msgs/String  {"w","h","dets":[{cls,conf,xyxy}...]}
  <out_ns>/image       sensor_msgs/Image annotated BGR
"""
import json
import threading
from pathlib import Path

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

import tensorrt as trt
import pycuda.driver as cuda

_TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# TRT 8.5 trt.nptype() touches the removed np.bool alias -> map by hand.
_TRT_TO_NP = {
    trt.DataType.FLOAT: np.float32,
    trt.DataType.HALF: np.float16,
    trt.DataType.INT8: np.int8,
    trt.DataType.INT32: np.int32,
    trt.DataType.BOOL: np.bool_,
}


def _letterbox(img, new_shape, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_shape - nh) // 2
    bot = new_shape - nh - top
    left = (new_shape - nw) // 2
    right = new_shape - nw - left
    padded = cv2.copyMakeBorder(resized, top, bot, left, right,
                                cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top)


def _nms(boxes, scores, iou_thr):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


class YoloTRT(object):
    """Self-contained TensorRT YOLO. CUDA context push/pop around inference so
    it is safe to call from a rospy subscriber thread (serialised by caller)."""

    def __init__(self, engine_path, device_id=0):
        cuda.init()
        self.cuda_ctx = cuda.Device(device_id).retain_primary_context()
        self.cuda_ctx.push()
        try:
            with open(engine_path, "rb") as f, trt.Runtime(_TRT_LOGGER) as rt:
                self.engine = rt.deserialize_cuda_engine(f.read())
            self.ctx = self.engine.create_execution_context()
            self.bindings = []
            for i in range(self.engine.num_bindings):
                shape = self.engine.get_binding_shape(i)
                dtype = _TRT_TO_NP[self.engine.get_binding_dtype(i)]
                host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
                dev = cuda.mem_alloc(host.nbytes)
                self.bindings.append(int(dev))
                if self.engine.binding_is_input(i):
                    self.in_shape, self.in_dtype = tuple(shape), dtype
                    self.h_in, self.d_in = host, dev
                else:
                    self.out_shape = tuple(shape)
                    self.h_out, self.d_out = host, dev
            self.imgsz = self.in_shape[2]      # square NCHW
            self.stream = cuda.Stream()
        finally:
            self.cuda_ctx.pop()

    def infer(self, bgr, conf_thr, iou_thr):
        H, W = bgr.shape[:2]
        img, r, pad = _letterbox(bgr, self.imgsz)
        img = img[:, :, ::-1].transpose(2, 0, 1)        # BGR HWC -> RGB CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        if self.in_dtype == np.float16:
            img = img.astype(np.float16)

        self.cuda_ctx.push()
        try:
            np.copyto(self.h_in, img.ravel())
            cuda.memcpy_htod_async(self.d_in, self.h_in, self.stream)
            self.ctx.execute_async_v2(bindings=self.bindings,
                                      stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
            self.stream.synchronize()
            out = np.array(self.h_out, copy=True).reshape(
                self.out_shape).astype(np.float32)
        finally:
            self.cuda_ctx.pop()
        return self._post(out, conf_thr, iou_thr, r, pad, (H, W))

    @staticmethod
    def _post(out, conf_thr, iou_thr, ratio, pad, orig):
        if out.ndim == 3:
            out = out[0]
        if out.shape[0] < out.shape[1]:
            out = out.T                                  # (N, 4+nc)
        xywh, scores = out[:, :4], out[:, 4:]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(scores)), cls]
        m = conf >= conf_thr
        if not m.any():
            return []
        xywh, cls, conf = xywh[m], cls[m], conf[m]
        x, y, w, h = xywh.T
        px, py = pad
        x1 = (x - w / 2 - px) / ratio
        y1 = (y - h / 2 - py) / ratio
        x2 = (x + w / 2 - px) / ratio
        y2 = (y + h / 2 - py) / ratio
        H, W = orig
        x1 = np.clip(x1, 0, W - 1); x2 = np.clip(x2, 0, W - 1)
        y1 = np.clip(y1, 0, H - 1); y2 = np.clip(y2, 0, H - 1)
        boxes = np.stack([x1, y1, x2, y2], 1)
        keep = _nms(boxes, conf, iou_thr)
        return [(int(cls[k]), float(conf[k]),
                 [float(boxes[k][0]), float(boxes[k][1]),
                  float(boxes[k][2]), float(boxes[k][3])]) for k in keep]


class YoloDelivery(object):
    def __init__(self):
        rospy.init_node("yolo_delivery")
        self.bridge = CvBridge()

        self.engine_path = rospy.get_param(
            "~engine", "/home/team5/EE478_final/best.engine")
        self.conf = float(rospy.get_param("~conf", 0.8))
        self.iou = float(rospy.get_param("~iou", 0.45))
        self.device_id = int(rospy.get_param("~gpu_device", 0))
        self.names = rospy.get_param(
            "~class_names", ["store", "pharmacy", "hamburger", "cafe"])
        in_topic = rospy.get_param("~image_topic", "/delivery/down_rgb")
        out_ns = rospy.get_param("~out_ns", "/delivery/yolo").rstrip("/")

        if not Path(self.engine_path).exists():
            rospy.logfatal("[yolo] engine not found: %s -- build it with "
                           "`yolo export model=best.pt format=engine "
                           "half=True imgsz=320`", self.engine_path)
            raise SystemExit(1)
        rospy.loginfo("[yolo] loading TensorRT engine %s ...", self.engine_path)
        self.trt = YoloTRT(self.engine_path, self.device_id)
        rospy.loginfo("[yolo] GPU engine ready (imgsz=%d, classes=%s)",
                      self.trt.imgsz, self.names)

        self.pub_det = rospy.Publisher(out_ns + "/detections", String,
                                       queue_size=5)
        self.pub_img = rospy.Publisher(out_ns + "/image", Image, queue_size=2)
        self._busy = threading.Lock()

        # Run inference only while ACTIVE (saves GPU). Subscribing straight to
        # the RAW camera colour topic (published from boot by the camera driver
        # on its own cores) and gating HERE -- instead of reading a relay that
        # gates the TOPIC -- means frames are already flowing the instant
        # /delivery/active flips, so YOLO starts with no relay-startup lag.
        self.gate_active = bool(rospy.get_param("~gate_active", False))
        self.rate_hz = float(rospy.get_param("~rate_hz", 0.0))   # 0 = no cap
        self._last_t = rospy.Time(0)
        # Always track /delivery/active so the image-watchdog knows when frames
        # SHOULD be flowing (no false 'no image' alarms before the trigger).
        # Inference is only HARD-gated on it when gate_active is set.
        self.active = False
        rospy.Subscriber(rospy.get_param("~active_topic", "/delivery/active"),
                         Bool, self.on_active, queue_size=1)

        # --- image-flow / inference health logging ---
        self.in_topic = in_topic
        self._n_in = 0           # total frames received on in_topic
        self._n_inf = 0          # total inferences run
        self._last_frame_t = None
        self._stat_t = rospy.Time(0)
        self._stat_in0 = 0
        self._inf_win = 0        # inferences this stats window
        self._det_win = 0        # frames-with-detection this window
        rospy.Subscriber(in_topic, Image, self.on_img, queue_size=1,
                         buff_size=2 ** 22)
        rospy.Timer(rospy.Duration(2.0), self._watchdog)
        rospy.loginfo("[yolo] %s -> %s (conf>=%.2f, GPU TRT, gate_active=%s, "
                      "rate=%.0f)", in_topic, out_ns, self.conf,
                      self.gate_active, self.rate_hz)

    def on_active(self, msg):
        was = self.active
        self.active = bool(msg.data)
        if self.active and not was:
            rospy.loginfo("[%s] ACTIVE -> inference ON", rospy.get_name())

    def _watchdog(self, _evt):
        """Warn if we are ACTIVE but no image frames are arriving -- this is the
        'is YOLO actually getting images?' check."""
        if not self.active:
            return                                  # delivery idle: no frames expected
        now = rospy.Time.now()
        if self._last_frame_t is None:
            rospy.logwarn("[%s] NO image yet on %s (active=%s) -- stream not "
                          "reaching YOLO", rospy.get_name(), self.in_topic,
                          self.active)
        elif (now - self._last_frame_t).to_sec() > 2.0:
            rospy.logwarn("[%s] image STALLED on %s (%.1fs since last frame, "
                          "active=%s)", rospy.get_name(), self.in_topic,
                          (now - self._last_frame_t).to_sec(), self.active)

    def _name(self, cid):
        return self.names[cid] if 0 <= cid < len(self.names) else str(cid)

    def on_img(self, msg):
        # count EVERY arriving frame (before any gate) so the stats reflect
        # whether images reach YOLO at all, independent of active/rate gating.
        self._n_in += 1
        now = rospy.Time.now()
        self._last_frame_t = now
        if self._stat_t.is_zero():
            self._stat_t = now
            rospy.loginfo("[%s] FIRST image on %s (%dx%d) -- stream OK",
                          rospy.get_name(), self.in_topic, msg.width, msg.height)
        elif (now - self._stat_t).to_sec() >= 2.0:
            dt = (now - self._stat_t).to_sec()
            rospy.loginfo("[%s] img in=%.1f Hz | inferred=%d dets=%d "
                          "(active=%s)", rospy.get_name(),
                          (self._n_in - self._stat_in0) / dt, self._inf_win,
                          self._det_win, self.active)
            self._stat_t, self._stat_in0 = now, self._n_in
            self._inf_win = self._det_win = 0

        if self.gate_active and not self.active:      # idle until active
            return
        if self.rate_hz > 0.0:                        # cap inference rate
            if (now - self._last_t).to_sec() < 1.0 / self.rate_hz:
                return
            self._last_t = now
        if not self._busy.acquire(blocking=False):    # drop while busy
            return
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            res = self.trt.infer(bgr, self.conf, self.iou)
            self._n_inf += 1
            self._inf_win += 1
            dets = [{"cls": self._name(c), "conf": round(cf, 3),
                     "xyxy": [round(v, 1) for v in xy]} for c, cf, xy in res]
            if dets:
                self._det_win += 1
            h, w = bgr.shape[:2]
            self.pub_det.publish(String(data=json.dumps(
                {"w": w, "h": h, "dets": dets})))
            if dets:
                rospy.loginfo_throttle(1.0, "[%s] DETECT %s", rospy.get_name(),
                                       [(d["cls"], d["conf"]) for d in dets])
            for c, cf, xy in res:
                x1, y1, x2, y2 = [int(v) for v in xy]
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(bgr, "%s %.2f" % (self._name(c), cf),
                            (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)
            out = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
            out.header = msg.header
            self.pub_img.publish(out)
        except Exception as e:
            rospy.logwarn_throttle(5.0, "[yolo] %s", e)
        finally:
            self._busy.release()


if __name__ == "__main__":
    try:
        YoloDelivery()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
