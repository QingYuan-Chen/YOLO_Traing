"""Run the verified YOLOv5n 320 plastic-bottle KModel on CanMV/K230.

Model contract:
    input:  uint8 [1, 3, 320, 320]
    output: float32 [1, 6300, 6]

The output has already been decoded by the exported YOLOv5 graph. It must be
paired with aidemo.yolov5_det_postprocess; scripts for YOLOv8/YOLO11 are not
interchangeable with this model.
"""

from libs.PipeLine import PipeLine, ScopedTiming  # type: ignore
from libs.AIBase import AIBase  # type: ignore
from libs.AI2D import Ai2d  # type: ignore
import os  # type: ignore
import gc  # type: ignore
import utime  # type: ignore
from media.media import *  # type: ignore
import nncase_runtime as nn  # type: ignore
import ulab.numpy as np  # type: ignore
import aidemo  # type: ignore


# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------
DISPLAY_MODE = "lcd"  # "lcd" or "hdmi"
RGB888P_SIZE = [1920, 1080]
DISPLAY_SIZE = [1920, 1080] if DISPLAY_MODE == "hdmi" else [640, 480]

KMODEL_PATH = "/sdcard/yolov5n_bottle_320_k230.kmodel"
MODEL_INPUT_SIZE = [320, 320]
LABELS = ["plastic-bottle"]

CONFIDENCE_THRESHOLD = 0.50
NMS_THRESHOLD = 0.45
MAX_BOXES_NUM = 50
EXPECTED_PREDICTIONS = 6300

SHOW_FPS = True
DEBUG_MODE = 0  # Set to 1 to print timing for each pipeline stage.


def require_file(path):
    """Fail before media initialization when the KModel path is wrong."""
    try:
        os.stat(path)
    except OSError:
        raise RuntimeError(
            "KModel not found: {}. Copy it to the SD card or update KMODEL_PATH.".format(
                path
            )
        )


class YOLOv5DetectionApp(AIBase):
    """CanMV video detector for one decoded YOLOv5 output tensor."""

    def __init__(
        self,
        kmodel_path,
        model_input_size,
        labels,
        confidence_threshold=0.5,
        nms_threshold=0.45,
        max_boxes_num=50,
        expected_predictions=6300,
        rgb888p_size=None,
        display_size=None,
        debug_mode=0,
    ):
        if rgb888p_size is None:
            rgb888p_size = [1920, 1080]
        if display_size is None:
            display_size = [640, 480]
        if len(labels) == 0:
            raise ValueError("LABELS must contain at least one class name.")
        if confidence_threshold < 0 or confidence_threshold > 1:
            raise ValueError("CONFIDENCE_THRESHOLD must be between 0 and 1.")
        if nms_threshold < 0 or nms_threshold > 1:
            raise ValueError("NMS_THRESHOLD must be between 0 and 1.")

        # AIBase creates nn.kpu(), loads the KModel and runs inference.
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.labels = labels
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_boxes_num = max_boxes_num
        self.expected_predictions = expected_predictions
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]  # type: ignore
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]  # type: ignore
        self.debug_mode = debug_mode
        self.output_checked = False

        inputs_num = self.kpu.inputs_size()
        outputs_num = self.kpu.outputs_size()
        if inputs_num != 1 or outputs_num != 1:
            raise RuntimeError(
                "YOLOv5 KModel contract mismatch: expected 1 input and 1 output, "
                "got {} input(s) and {} output(s).".format(inputs_num, outputs_num)
            )

        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(
            nn.ai2d_format.NCHW_FMT,
            nn.ai2d_format.NCHW_FMT,
            np.uint8,
            np.uint8,
        )

        print("[KPU] loaded: {}".format(self.kmodel_path))
        print(
            "[KPU] expected input=uint8 [1, 3, {}, {}], output=float32 "
            "[1, {}, {}]".format(
                self.model_input_size[1],
                self.model_input_size[0],
                self.expected_predictions,
                5 + len(self.labels),
            )
        )

    def _letterbox_padding(self, input_size):
        src_w, src_h = input_size
        dst_w, dst_h = self.model_input_size
        scale = min(dst_w / src_w, dst_h / src_h)
        new_w = int(src_w * scale)
        new_h = int(src_h * scale)

        # This intentionally matches CanMV's letterbox_pad_param: the resized
        # image starts at (0, 0), and padding is added only on the right/bottom.
        # aidemo.yolov5_det_postprocess uses the same coordinate convention.
        top = 0
        bottom = dst_h - new_h
        left = 0
        right = dst_w - new_w
        return top, bottom, left, right, scale

    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = (
                input_image_size if input_image_size else self.rgb888p_size
            )
            top, bottom, left, right, scale = self._letterbox_padding(
                ai2d_input_size
            )
            print(
                "[AI2D] source={}x{}, model={}x{}, scale={:.6f}, "
                "padding=(top {}, bottom {}, left {}, right {})".format(
                    ai2d_input_size[0],
                    ai2d_input_size[1],
                    self.model_input_size[0],
                    self.model_input_size[1],
                    scale,
                    top,
                    bottom,
                    left,
                    right,
                )
            )
            self.ai2d.pad(
                [0, 0, 0, 0, top, bottom, left, right],
                0,
                [128, 128, 128],
            )
            self.ai2d.resize(
                nn.interp_method.tf_bilinear,
                nn.interp_mode.half_pixel,
            )
            self.ai2d.build(
                [1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                [1, 3, self.model_input_size[1], self.model_input_size[0]],
            )

    def _check_output_contract(self, output):
        if self.output_checked:
            return

        shape = output.shape
        expected_columns = 5 + len(self.labels)
        if (
            len(shape) != 3
            or shape[0] != 1
            or shape[1] != self.expected_predictions
            or shape[2] != expected_columns
        ):
            raise RuntimeError(
                "YOLOv5 output mismatch: expected [1, {}, {}], got {}. "
                "Check the KModel, MODEL_INPUT_SIZE and LABELS.".format(
                    self.expected_predictions,
                    expected_columns,
                    shape,
                )
            )

        print("[KPU] output[0] shape verified: {}".format(shape))
        self.output_checked = True

    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            if len(results) != 1:
                raise RuntimeError(
                    "YOLOv5 KModel output mismatch: expected 1 output, got {}".format(
                        len(results)
                    )
                )

            self._check_output_contract(results[0])
            return aidemo.yolov5_det_postprocess(
                results[0][0],
                [self.rgb888p_size[1], self.rgb888p_size[0]],
                [self.model_input_size[1], self.model_input_size[0]],
                [self.display_size[1], self.display_size[0]],
                len(self.labels),
                self.confidence_threshold,
                self.nms_threshold,
                self.max_boxes_num,
            )

    def draw_result(self, pl, result):
        with ScopedTiming("display draw", self.debug_mode > 0):
            pl.osd_img.clear()
            if not result:
                return

            boxes, class_ids, scores = result
            for i in range(len(boxes)):
                class_id = int(class_ids[i])
                if class_id < 0 or class_id >= len(self.labels):
                    print("[WARN] ignored invalid class id: {}".format(class_id))
                    continue

                x, y, w, h = map(lambda value: int(round(value, 0)), boxes[i])
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(self.display_size[0], x + w)
                y2 = min(self.display_size[1], y + h)
                x, y = x1, y1
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue

                pl.osd_img.draw_rectangle(
                    x,
                    y,
                    w,
                    h,
                    color=(255, 255, 0, 0),
                    thickness=2,
                )
                pl.osd_img.draw_string_advanced(
                    x,
                    max(0, y - 36),
                    32,
                    "{} {:.2f}".format(self.labels[class_id], scores[i]),
                    color=(255, 255, 0, 0),
                )


if __name__ == "__main__":
    pl = None
    yolo_det = None
    try:
        require_file(KMODEL_PATH)
        os.exitpoint(os.EXITPOINT_ENABLE)

        pl = PipeLine(
            rgb888p_size=RGB888P_SIZE,
            display_size=DISPLAY_SIZE,
            display_mode=DISPLAY_MODE,
            debug_mode=DEBUG_MODE,
        )
        pl.create()

        yolo_det = YOLOv5DetectionApp(
            KMODEL_PATH,
            model_input_size=MODEL_INPUT_SIZE,
            labels=LABELS,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            nms_threshold=NMS_THRESHOLD,
            max_boxes_num=MAX_BOXES_NUM,
            expected_predictions=EXPECTED_PREDICTIONS,
            rgb888p_size=RGB888P_SIZE,
            display_size=DISPLAY_SIZE,
            debug_mode=DEBUG_MODE,
        )
        yolo_det.config_preprocess()

        last_time = utime.ticks_ms()
        fps = 0.0
        while True:
            os.exitpoint()
            with ScopedTiming("total", DEBUG_MODE > 0):
                img = pl.get_frame()
                result = yolo_det.run(img)
                yolo_det.draw_result(pl, result)

                current_time = utime.ticks_ms()
                elapsed_ms = utime.ticks_diff(current_time, last_time)
                last_time = current_time
                if elapsed_ms > 0:
                    current_fps = 1000.0 / elapsed_ms
                    fps = current_fps if fps == 0 else fps * 0.9 + current_fps * 0.1

                if SHOW_FPS:
                    pl.osd_img.draw_string_advanced(
                        20,
                        20,
                        32,
                        "FPS: {:.1f}".format(fps),
                        color=(255, 255, 0, 0),
                    )

                pl.show_image()
                gc.collect()
    except Exception as error:
        print("[ERROR] {}".format(error))
        raise
    finally:
        if yolo_det is not None:
            try:
                yolo_det.deinit()
            except Exception as error:
                print("[WARN] detector cleanup failed: {}".format(error))
        if pl is not None:
            try:
                pl.destroy()
            except Exception as error:
                print("[WARN] pipeline cleanup failed: {}".format(error))
