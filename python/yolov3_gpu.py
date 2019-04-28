from ctypes import *
import math
import os
import numpy as np

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


def load_yolo_dll(dll_name):
    cwd = os.path.dirname(__file__)
    dll_path = os.path.join(cwd, dll_name)
    lib = CDLL(dll_path, RTLD_GLOBAL)

    ### setup dll interfaces
    # int network_width(network *net);
    if hasattr(lib, 'network_width'):
        lib.network_width.argtypes = [c_void_p]
        lib.network_width.restype = c_int

    # int network_height(network *net);
    if hasattr(lib, 'network_height'):
        lib.network_height.argtypes = [c_void_p]
        lib.network_height.restype = c_int
    
    # int network_channel(network *net);
    if hasattr(lib, 'network_channel'):
        lib.network_channel.argtypes = [c_void_p]
        lib.network_channel.restype = c_int

    # int network_batch(network *net);
    if hasattr(lib, 'network_batch'):
        lib.network_batch.argtypes = [c_void_p]
        lib.network_batch.restype = c_int

    # void cuda_set_device(int n);
    if hasattr(lib, 'cuda_set_device'):
        lib.cuda_set_device.argtypes = [c_int]

    # float *network_predict(network net, float *input);
    if hasattr(lib, 'network_predict'):
        lib.network_predict.argtypes = [c_void_p, POINTER(c_float)]
        lib.network_predict.restype = POINTER(c_float)

    # image make_image(int w, int h, int c);
    if hasattr(lib, 'make_image'):
        lib.make_image.argtypes = [c_int, c_int, c_int]
        lib.make_image.restype = IMAGE

    # detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter);
    if hasattr(lib, 'get_network_boxes'):
        lib.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
        lib.get_network_boxes.restype = POINTER(DETECTION)

    # detection *get_network_boxes_batch(network *net, int b, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter);
    if hasattr(lib, 'get_network_boxes_batch'):
        lib.get_network_boxes_batch.argtypes = [c_void_p, c_int, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
        lib.get_network_boxes_batch.restype = POINTER(DETECTION)

    # detection *make_network_boxes(network *net, float thresh, int *num);
    if hasattr(lib, 'make_network_boxes'):
        lib.make_network_boxes.argtypes = [c_void_p, c_float, POINTER(c_int)]
        lib.make_network_boxes.restype = POINTER(DETECTION)

    # detection *make_network_boxes_batch(network *net, int b, float thresh, int *num);
    if hasattr(lib, 'make_network_boxes_batch'):
        lib.make_network_boxes_batch.argtypes = [c_void_p, c_int, c_float, POINTER(c_int)]
        lib.make_network_boxes_batch.restype = POINTER(DETECTION)

    # void free_detections(detection *dets, int n);
    if hasattr(lib, 'free_detections'):
        lib.free_detections.argtypes = [POINTER(DETECTION), c_int]

    # void free_ptrs(void **ptrs, int n);
    if hasattr(lib, 'free_ptrs'):
        lib.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

    # float *network_predict(network net, float *input);
    if hasattr(lib, 'network_predict'):
        lib.network_predict.argtypes = [c_void_p, POINTER(c_float)]
        lib.network_predict.restype = POINTER(c_float)

    # float *network_predict_image(network *net, image im);
    if hasattr(lib, 'network_predict_image'):
        lib.network_predict_image.argtypes = [c_void_p, IMAGE]
        lib.network_predict_image.restype = POINTER(c_float)

    # network *load_network_custom(char *cfg, char *weights, int clear, int batch);
    if hasattr(lib, 'load_network_custom'):
        lib.load_network_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
        lib.load_network_custom.restype = c_void_p

    # network *load_network_custom_mem(char *cfg_buffer, char *weight_buffer, int clear, int batch);
    if hasattr(lib, 'load_network_custom_mem'):
        lib.load_network_custom_mem.argtypes = [c_char_p, c_char_p, c_int, c_int]
        lib.load_network_custom_mem.restype = c_void_p

    # network *load_network_custom_one(char *cfg_weights, int clear, int batch);
    if hasattr(lib, 'load_network_custom_one'):
        lib.load_network_custom_one.argtypes = [c_char_p, c_int, c_int]
        lib.load_network_custom_one.restype = c_void_p

    # void do_nms_sort(detection *dets, int total, int classes, float thresh);
    if hasattr(lib, 'do_nms_sort'):
        lib.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

    # void do_nms_obj(detection *dets, int total, int classes, float thresh);
    if hasattr(lib, 'do_nms_obj'):
        lib.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

    # void free_image(image m);
    if hasattr(lib, 'free_image'):
        lib.free_image.argtypes = [IMAGE]

    # image letterbox_image(image im, int w, int h); 
    if hasattr(lib, 'letterbox_image'):
        lib.letterbox_image.argtypes = [IMAGE, c_int, c_int]
        lib.letterbox_image.restype = IMAGE

    # image resize_image(image im, int w, int h);
    if hasattr(lib, 'resize_image'):
        lib.resize_image.argtypes = [IMAGE, c_int, c_int]
        lib.resize_image.restype = IMAGE

    # metadata get_metadata(char *file);
    if hasattr(lib, 'get_metadata'):
        lib.get_metadata.argtypes = [c_char_p]
        lib.get_metadata.restype = METADATA

    # image load_image_color(char *filename, int w, int h);
    if hasattr(lib, 'load_image_color'):
        lib.load_image_color.argtypes = [c_char_p, c_int, c_int]
        lib.load_image_color.restype = IMAGE

    # void rgbgr_image(image im);
    if hasattr(lib, 'rgbgr_image'):
        lib.rgbgr_image.argtypes = [IMAGE]

    return lib

class YOLOParamsGPU:
    def __init__(self, batch, gpu=0, encrypt = False, nclasses = 3, bgr=False, thr=0.5, nms=0.45):
        self.batch = batch
        self.gpu = gpu
        self.bgr = bgr
        self.nclasses = nclasses
        self.confThreshold = thr
        self.nmsThreshold = nms
        self.width = 0
        self.height = 0
        self.channel = 0
        self.encrypt = encrypt

class YOLOModelGPU():
    def __init__(self, config_file, model_file, param):
        self.lib = load_yolo_dll("yolo_cpp_dll.dll")
        self.lib.cuda_set_device(param.gpu)

        import time
        st = time.perf_counter()
        param.encrypt = False

        print("加载检测模型文件: {} ...".format(model_file))
        if not param.encrypt:
            self.net = self.lib.load_network_custom(config_file.encode("utf-8"), model_file.encode("utf-8"), 0, param.batch)
        print("加载完成: {:.3f} sec".format(time.perf_counter() - st))

        self.param = param
        self.param.width = self.lib.network_width(self.net)
        self.param.height = self.lib.network_height(self.net)
        self.param.channel = self.lib.network_channel(self.net)
    
    def load_batch_images(self, paths):
        from opencv import cv2
        paths = paths[:self.param.batch]
        self.frames = []
        self.ori_frames = []
        for p in paths:
            im = self.lib.load_image_color(p.encode("utf-8"), 0, 0)
            self.frames.append(im)
            self.ori_frames.append(cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR))
        return self.frames
   
    def preprocess(self, frames):
        w = self.param.width
        h = self.param.height
        c = self.param.channel
        b = self.param.batch
        s = w * h * c
        X = (c_float * (s * b))()
        for i, im in enumerate(frames):
            sized = self.lib.resize_image(im, w, h)
            self.letterbox = 0
            X[i*s:(i+1)*s] = sized.data[0:s]
            self.lib.free_image(sized)
        return X

    def detect(self):
        self.lib.network_predict(self.net, self.preprocess(self.frames))
        num = c_int(0)
        pnum = pointer(num)
        results = []
        nclasses = self.param.nclasses
        thresh=self.param.confThreshold
        nms=self.param.nmsThreshold
        for n, im in enumerate(self.frames):
            dets = self.lib.get_network_boxes_batch(self.net, n, im.w, im.h, thresh, .5, None, 0, pnum, self.letterbox)
            num = pnum[0]
            self.lib.do_nms_sort(dets, num, nclasses, nms)
            result = []
            for j in range(num):
                for i in range(nclasses):
                    if dets[j].prob[i] > 0:
                        b = dets[j].bbox
                        result.append({"class": i, "confidence": dets[j].prob[i], "box": (
                            int(b.x - b.w/2), int(b.y - b.h/2), int(b.w), int(b.h))})
            result = sorted(result, key=lambda x: -x["confidence"])
            self.lib.free_detections(dets, num)
            results.append(result)
        return results

    def postprocess(self, outs):
        results = []
        if self.ori_frames:
            for frame, result in zip(self.ori_frames, outs):
                results.append({"frame": frame, "result": result})
        else:
            for result in outs:
                results.append({"frame": None, "result": result})
        # free images
        for im in self.frames:
            self.lib.free_image(im)
            
        return results

    def run_batch_paths(self, paths):
        # call the load image interface from dll
        self.load_batch_images(paths)
        outs = self.detect()
        results = self.postprocess(outs)
        return results
    
    def run_batch_paths2(self, paths):
        import time
        paths = paths[:self.param.batch]
        frames = []
        st = time.perf_counter()
        for p in paths:
            frame = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
            frames.append(frame)
        print("load: {} ms".format(time.perf_counter() - st))
        return self.run_batch(frames)
    
    def run_batch(self, frames):
        import time
        def array_to_image(arr):
            narr = np.array(arr)
            narr = narr.transpose(2,0,1)
            c = narr.shape[0]
            h = narr.shape[1]
            w = narr.shape[2]
            narr = np.ascontiguousarray(narr.flat, dtype=np.float32) / 255.0            
            data = (c_float * (w*h*c))()
            data[0:w*h*c] = narr.ctypes.data_as(POINTER(c_float))[0:w*h*c]
            im = IMAGE(w,h,c,data)
            return im

        st = time.perf_counter()
        self.frames = [array_to_image(frame) for frame in frames]
        self.ori_frames = frames
        print("process: {} s".format(time.perf_counter() - st))
        outs = self.detect()
        results = self.postprocess(outs)
        return results

if __name__ == "__main__":
    import time
    from opencv import cv2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    yoloParam = YOLOParamsGPU(4, 0, encrypt = False, nclasses=4, thr=0.5)
    yoloModel = YOLOModelGPU("models/yolo/cfg/my-yolov3.cfg.enc", "models/yolo/my-yolov3_final.weights.enc", yoloParam)

    st = time.perf_counter()
    results = yoloModel.run_batch([cv2.imread(str(i) + ".jpg") for i in range(1, 5)])
    print("batch[{}]: {:.3f} s".format(len(results), time.perf_counter() - st))
    for res in results:
        print(res["result"])
        for r in res["result"]:                
            x,y,w,h = r["box"]
            cv2.rectangle(res["frame"], (x, y), (x+w, y+h), (0, 255, 0))
        if res["frame"].shape[0] > 1024:
            res["frame"] = cv2.resize(res["frame"], (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("frame", res["frame"])
        cv2.waitKey()
    cv2.destroyAllWindows()
