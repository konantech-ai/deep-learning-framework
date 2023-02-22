from .dataset import Dataset
from ..tensor import Tensor
from typing import Callable, Optional

import os
import cv2
import time
import random
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

class CocoDataset(Dataset):
    static_tensors = None
    
    _image_size = 608

    def __init__(self, session, name: str, train: bool = True, download: bool = False,
                 data_root: str or None = "data", cache_root: str or None = "cache", 
                 transform: Optional[Callable] = None, verbose: bool = False, select: str = "random",
                 letterSize: bool=False, args: Optional[dict] = {}):        
        # Issue (2023-01-11) : transform 미구현
        super(CocoDataset, self).__init__(session, name, train, download, data_root, cache_root, transform, verbose, select, args)
        
        self.letterSize = letterSize;

        # set data using the method of parent class 'Dataset'
        self.setData()
        self.createConstants()

        self.categories = self.ann_map["categories"]
        self.image_path = self.ann_map["image_path"]
        self.image_info_list = self.ann_map["image_info_list"]

        self.canvas = None
        self.proc_count = 0
        self.time = time.time()

    def __del__(self):
        pass

    @property
    def len(self) -> int:
        return len(self.ann_map["image_info_list"])

    def load_data(self, data_path:str, train:bool) -> None:
        image_path = data_path + ("/train2014/" if train else "/val2014/")

        mode = "train" if train else "test"
        verbose = self.verbose

        if verbose: print(f"loading yolo4 information files for {mode} mode...", )

        filename = "instances_train2014.json" if train else "instances_val2014.json"
        inst_path = data_path + "/annotations_trainval2014/annotations/" + filename;

        if verbose: print("   parsing annation file...");

        inst_info = self.session.parse_json_file(inst_path);

        if verbose: print("   setting  category information...");

        categories = {}
        for cat_term in inst_info["categories"]:
            categories[str(cat_term["id"])] = cat_term["name"]

        lost_file = 0
        nth_file = 0

        image_info_map = {}
        image_info_list = inst_info["images"]

        if verbose: print(f"   checking file information: {len(image_info_list)} files...")

        for image_term in image_info_list:
            filepath = image_path + image_term["file_name"]
            if os.path.exists(filepath):
                image_id = str(image_term["id"])
                image_info = {}
                image_info["image_id"] = image_id;
                image_info["width"] = image_term["width"]
                image_info["height"] = image_term["height"]
                image_info["file_name"] = image_term["file_name"]
                image_info["annotations"] = []

                image_info_map[image_id] = image_info;
            else:
                lost_file += 1;

            nth_file += 1
            if nth_file % 5000 == 0:
                if verbose: print(f"      {nth_file}/{len(image_info_list)} files processed");

        if lost_file > 0:
            if verbose: print(f"      => there are {lost_file} missing files among {len(image_info_list)} files");
        else:
            if verbose: print(f"      => there is no missing file among {len(image_info_list)} files");

        image_ann_list = inst_info["annotations"]

        nth_image = 0
        lost_info = 0

        print(f"   extracting annotation informatiion: {len(image_ann_list)} images...");

        for ann_term in image_ann_list:
            image_id = str(ann_term["image_id"])

            if image_id in image_info_map.keys():
                ann_info = {}

                ann_info["category"] = ann_term["category_id"]
                ann_info["bbox"] = ann_term["bbox"]

                image_info = image_info_map[image_id];
                annotations = image_info["annotations"];
                annotations.append(ann_info);
            else:
                lost_info += 1

            nth_image += 1
            if nth_image % 500 == 0:
                if verbose: print(f"      {nth_image}/{len(image_ann_list)} files processed")

        if lost_info > 0:
            if verbose: print(f"      => there are {lost_info} images missing information among {len(image_ann_list)} images");
        else:
            if verbose: print(f"      => there is no image missing information among {len(image_ann_list)} files");

        if verbose: print("   keeping annotation informatiion...");

        image_infos = list(image_info_map.values())
        self.ann_map = {"categories":categories, "image_path":image_path, "image_info_list":image_infos}

    def save_cache(self, cachePath:str, train:bool) -> None:
        if not self.get_arg("save_cache", True): return

        mode = "train" if train else "test"
        verbose = self.verbose

        if verbose: print("saving yolo cache data...")

        postfix = ".train" if train  else ".test"
        filepath = cachePath + '/' + self.name + postfix + ".dat"
        self.session.save_data(self.ann_map, filepath);

        if verbose: print("yolo cache data was saved")
        
    def load_cache(self, cachePath:str, train:bool) -> bool:
        try:
            if not self.get_arg("load_cache", True): return False

            mode = "train" if train else "test"
            verbose = self.verbose

            if verbose: print(f"loading yolo cache data for {mode} mode...")

            postfix = ".train" if train  else ".test"
            filepath = cachePath + '/' + self.name + postfix + ".dat"

            if not os.path.exists(filepath):
                return False

            self.ann_map = self.session.load_data(filepath);

            if verbose: print(f"yolo cache data was loaded for {mode} mode")

            return True
        except:
            return False

    def createConstants(self):
        self.image_size = 608

        self.scale = 3
        self.anc_per_scale = 3

        self.class_num = 80
        self.true_vec_size = 92   # 4 for bounding-box, 1 for confidence, 80 for classes, 7 extra information

        self.grid_cnt = [76, 38, 19]
        self.grid_size = [8, 16, 32]

        self.grid_cell_count = (76 * 76) + (38 * 38) + (19 * 19)

        self.anchor = [
            [12, 16, 19, 36, 40, 28],
            [36, 75, 76, 55, 72, 146],
            [142, 110, 192, 243, 459, 401]]

        self.max_true_box_cnt = 50

        self.iou_normalizer = 0.07
        self.poor_iou_thresh = 0.70 # 0.213 # 0.70
        self.near_iou_thresh = 0.80
        self.class_prob_thresh = 0.25
        
        if CocoDataset.static_tensors is not None:
            return

        grid_size         = self.session.createTensor((self.grid_cell_count * 3, 2), "float32")
        offset            = self.session.createTensor((self.grid_cell_count * 3, 2), "float32")
        anchor            = self.session.createTensor((self.grid_cell_count * 3, 2), "float32")
        image_size        = self.session.createTensor((1,), "float32")
        iou_normalizer    = self.session.createTensor((1,), "float32")
        poor_iou_thresh   = self.session.createTensor((1,), "float32")
        near_iou_thresh   = self.session.createTensor((1,), "float32")
        class_prob_thresh = self.session.createTensor((1,), "float32")

        ng = 0

        for ns in range(self.scale):
            for n in range(self.grid_cnt[ns]):
                for m in range(self.grid_cnt[ns]):
                    for k in range(3):
                        grid_size[ng * 3 + k, 0] = float(self.grid_size[ns])
                        grid_size[ng * 3 + k, 1] = float(self.grid_size[ns])

                        offset[ng * 3 + k, 0] = float(m)
                        offset[ng * 3 + k, 1] = float(n)

                        anchor[ng * 3 + k, 0] = float(self.anchor[ns][k * 2 + 0])
                        anchor[ng * 3 + k, 1] = float(self.anchor[ns][k * 2 + 1])
                    ng += 1

        image_size[0]        = self.image_size
        iou_normalizer[0]    = self.iou_normalizer
        poor_iou_thresh[0]   = self.poor_iou_thresh
        near_iou_thresh[0]   = self.near_iou_thresh
        class_prob_thresh[0] = self.class_prob_thresh
        
        CocoDataset.static_tensors = {
            "grid_size":grid_size, "offset":offset,
            "anchor":anchor, "image_size":image_size,
            "iou_normalizer":iou_normalizer, "poor_iou_thresh":poor_iou_thresh, 
            "near_iou_thresh":near_iou_thresh, "class_prob_thresh":class_prob_thresh }

    def getFieldInfo(self, is_x:bool) -> dict:
        if is_x:
            return {"#": {"shape":(self.image_size, self.image_size, 3), "type":"float32"}}
        else:
            return {
                "true_box_cood": {"shape":(self.max_true_box_cnt, 4), "type":"float32"},
                "true_box_class": {"shape":(self.max_true_box_cnt, 1), "type":"int32"},
                "true_box_anchor_pos": {"shape":(self.max_true_box_cnt,), "type":"int32"}}


    def createAuxBuffer(self, batch_size):
        shape = (self.image_size, self.image_size, 3)
        imgBuf = self.session.createTensor(shape, "float32")
        return imgBuf

    def getDataPair(self, idxes, xsBuf, ysBuf, auxBuf):
        self.idxes = idxes

        xsBuf["#"].set_zero()

        if ysBuf is not None:
            ysBuf["true_box_cood"].set_zero()
            ysBuf["true_box_class"].set_zero()
            ysBuf["true_box_anchor_pos"].set_zero()

        for nth, idx1 in enumerate(idxes):
            idx2, mix_ratio, mixMode = self.select_mixup_image(idx1, nth, idxes)
            #print(f"idx1:{idx1}, idx2:{idx2}, mix_ratio:{mix_ratio}, mixMode:{mixMode}")

            image_info1, image_info2 = self.fill_image_pixels(idx1, idx2, mix_ratio, mixMode, xsBuf["#"], nth, auxBuf)

            #print("image_info1", image_info1)
            #print("image_info2", image_info2)

            self.fill_box_info(image_info1, image_info2, nth, ysBuf)

    def select_mixup_image(self, idx, nth, idxes):
        if random.random() < 1.0: #0.7:
            return -1, 1.0, 0   # no-mix

        seed = random.random()
        dice = beta(1.5, 1.5).pdf(seed)

        batch_size = len(idxes)
        idx2 = idxes[(nth+random.randint(1, batch_size-1))%batch_size]

        return idx2, dice/2, 1 # resolve

    def fill_image_pixels(self, idx1, idx2, mix_ratio, mixMode, x, nth, auxBuf):
        image_info1 = self.image_info_list[idx1]

        filepath = self.image_path + image_info1["file_name"]

        auxBuf.load_jpeg_pixels(filepath, transpose=False, code=cv2.COLOR_RGB2BGR, mix=1.0)

        if idx2 >= 0:
            image_info2 = self.image_info_list[idx2]

            filepath = self.image_path + image_info2["file_name"]
        
            auxBuf.load_jpeg_pixels(filepath, transpose=False, code=cv2.COLOR_RGB2BGR, mix=mix_ratio)
        else:
            image_info2 = None

        x.copy_into_row(nth, auxBuf)

        return image_info1, image_info2

    def fill_box_info(self, image_info1, image_info2, nth, ysBuf):
        pos = self.collect_box_info(image_info1, nth, ysBuf, 0)
        pos = self.collect_box_info(image_info2, nth, ysBuf, pos)
    
    def collect_box_info(self, image_info, nth, ysBuf, nfrom):
        if image_info is None:
            return nfrom

        annotations = image_info["annotations"]

        if len(annotations) == 0: return nfrom

        nto = nfrom + len(annotations)
        if nto > self.max_true_box_cnt:
            annotations = annotations[0:self.max_true_box_cnt-nfrom]
            nto = self.max_true_box_cnt

        image_width, image_height = image_info['width'], image_info['height']
        image_size = max(image_width, image_height)
        ratio = self.image_size / image_size
        size = np.array([image_width, image_height], dtype=np.float32)
        shift = (1 - size / image_size) * (self.image_size / 2)

        box_list = [term["bbox"] for term in annotations]
        boxes = np.array(box_list, dtype=np.float32).reshape([-1,4])    # reshape is needed for the case when there is no box
        boxes = boxes * ratio
        boxes[:,0:2] = boxes[:,0:2] + boxes[:,2:4]/ 2 + shift

        cats = [term["category"] for term in annotations]
        cats = np.array(cats, dtype=np.int32)

        ious = np.array([[self.iou_anchor(box, na) for na in range(9)] for box in boxes])

        best_anchors = ious.argmax(axis=1)
        best_pos = [self.seek_best_pos(n, best_anchors, boxes) for n in range(len(box_list))]
        best_pos = np.array(best_pos, dtype=np.int32)

        if ysBuf is not None:
            ysBuf["true_box_cood"][nth, nfrom:nto, :] = boxes
            ysBuf["true_box_class"][nth, nfrom:nto, :] = cats
            ysBuf["true_box_anchor_pos"][nth, nfrom:nto] = best_pos

        return nto

    def iou_anchor(self, box, n):
        ns, na = n // 3, n % 3
        anchor_width = self.anchor[ns][na*2+0]
        anchor_height = self.anchor[ns][na*2+1]
        
        max_width = max(box[2], anchor_width)
        min_width = min(box[2], anchor_width)
        
        max_height = max(box[3], anchor_height)
        min_height = min(box[3], anchor_height)
        
        intersect_area = min_height * min_width
        union_area = max_height * max_width
        
        iou = intersect_area / union_area  # ��ġ�� ��ģ�ٰ� �����ϰ� ũ�⸸ ����ϱ� �̷��� �����ϰ� ��� ����

        return iou

    def seek_best_pos(self, n, best_anchors, boxes):
        best_anchor = int(best_anchors[n])
        ns, na = best_anchor // 3, best_anchor % 3

        pos = 0

        if ns > 0: pos += 76 * 76 * 3
        if ns > 1: pos += 38 * 38 * 3

        grid_x = boxes[n][0] // self.grid_size[ns]
        grid_y = boxes[n][1] // self.grid_size[ns]

        pos += (grid_x * self.grid_cnt[ns] + grid_y) * 3 + na;

        return pos

    def show_inf_result(self, metrics, timeout, win_close, disp_size, disp_rows):
        self.names_path = "Z:\\kai_data\\yolo\\coco2014\\coco.names"
        categories = self.session.read_file_lines(self.names_path)

        batch_size = len(self.idxes)

        disp_width = (disp_size + 10) * ((batch_size - 1) // disp_rows + 1) + 10

        if self.has_annotation:
            disp_height = (disp_size + 10) * 2 * disp_rows + 10
        else:
            disp_height = (disp_size + 10) * disp_rows + 10

        if self.canvas is None:
            self.canvas = np.full(shape=(disp_height, disp_width, 3), fill_value=255, dtype=np.uint8)
        else:
            self.canvas[:] = 255

        for n in range(batch_size):
            image_info = self.image_info_list[self.idxes[n]]
            filepath = self.image_path + image_info["file_name"]
        
            img = cv2.imread(filepath)
            
            height, width, _ = img.shape
            ratio = disp_size / max(height, width)
            hratio = height * ratio / self.image_size
            wratio = width * ratio / self.image_size
            img_height = int(height * ratio)
            img_width = int(width * ratio)

            img1 = cv2.resize(img,(img_width, img_height))
            img2 = cv2.resize(img,(img_width, img_height))
            
            obj_confs = metrics["conf"][n]
            obj_classes = metrics["class"][n]
            obj_boxes = metrics["box"][n]

            for m in range(obj_confs.shape[0]):
                prob = obj_confs[m].item()
                if prob < 0.25: continue
                obj_class = obj_classes[m].item()
                box = obj_boxes[m]

                w = int(box[2].item() * wratio)
                h = int(box[3].item() * hratio)
                x = int(box[0].item() * wratio - w / 2)
                y = int(box[1].item() * hratio - h / 2)

                class_name = categories[int(obj_class)]
                color = (m//9*127, (m//3)%3*127, m%3*127)
                text = f"{class_name} {prob:.2f}"

                cv2.rectangle(img1,(x,y),(x+w,y+h),color, 2)
                cv2.putText(img1, text, (x+5, y+25), 5, 1, color);

            if self.has_annotation:
                annotations = image_info["annotations"]

                for m, term in enumerate(annotations):
                    box = term["bbox"]
                    cat = term["category"]

                    left = int(box[0] * ratio)
                    top = int(box[1] * ratio)
                    right = int((left + box[2]) * ratio)
                    bottom = int((top + box[3]) * ratio)

                    class_name = self.categories[str(cat)]
                    color = (m//9*127, (m//3)%3*127, m%3*127)
                    text = f"{class_name}"

                    cv2.rectangle(img2,(left, top),(right, bottom),color, 2)
                    cv2.putText(img2, text, (left+5, top+25), 5, 1, color)

            wbase = (disp_size - img_width) // 2 + 10 + (n // disp_rows) * (disp_size + 10)

            if self.has_annotation:
                hbase1 = (disp_size - img_height) // 2 + (disp_size * 2 + 10) * (n % disp_rows) + 10
                hbase2 = hbase1 + disp_size + 10
                self.canvas[hbase1:img_height+hbase1, wbase:img_width+wbase,:] = img1
                self.canvas[hbase2:img_height+hbase2, wbase:img_width+wbase,:] = img2
            else:
                hbase1 = (disp_size - img_height) // 2 + (disp_size + 10) * (n % disp_rows) + 10
                self.canvas[hbase1:img_height+hbase1, wbase:img_width+wbase,:] = img1

        cv2.imshow("yolo_result",self.canvas)
        cv2.waitKey(timeout)
        if win_close: cv2.destroyAllWindows()
        
        self.proc_count += batch_size
        time_diff = int(time.time() - self.time)
        print(f"    {self.proc_count} images processed ({time_diff} secs)")

class Coco2014(CocoDataset):
    def __init__(self, session, train: bool = True, download: bool = False,
                 data_root: str or None = "data", cache_root: str or None = "cache", 
                 transform: Optional[Callable] = None, verbose: bool = False, select: str = "random",
                 letterSize: bool=False, args: Optional[dict] = {}):        
        # Issue (2023-01-11) : transform �̱���
        super(Coco2014, self).__init__(session, "coco2014", train, download, data_root, cache_root, 
                                       transform, verbose, select, letterSize, args)
        self.has_annotation = True

class Coco2014Inference(CocoDataset):
    def __init__(self, session, data_root: str, data_filter: str, names_path: str, verbose: bool = False, args: Optional[dict] = {}):        
        self.data_filter = data_filter
        self.names_path = names_path
        super(Coco2014Inference, self).__init__(session, "", False, False, data_root, None, 
                                       None, verbose, "sequential", False, args)
        self.has_annotation = False

    def __del__(self):
        pass

    def load_data(self, data_path:str, train:bool) -> None:
        verbose = self.verbose

        if verbose: print(f"loading test data from {data_path}")

        fileinfos = []

        for filename in os.listdir(data_path):
            subpath = os.path.join(data_path, filename)
            if fnmatch.fnmatch(filename, self.data_filter) and os.path.isfile(subpath):
                fileinfos.append({"file_name":filename})

        categories = self.session.read_file_lines(self.names_path)
        image_path = data_path + "/"
        image_info_list = fileinfos

        self.ann_map = {"categories":categories, "image_path":image_path, "image_info_list":image_info_list}

    def getFieldInfo(self, is_x:bool) -> dict:
        if is_x:
            return {"#": {"shape":(self.image_size, self.image_size, 3), "type":"float32"}}
        else:
            return {}

    def getDataPair(self, idxes, xsBuf, ysBuf, auxBuf):
        self.idxes = idxes
        xsBuf["#"].set_zero()

        for nth, idx in enumerate(idxes):
            image_info, _ = self.fill_image_pixels(idx, -1, 0, 0, xsBuf["#"], nth, auxBuf)

    def show_inf_result(self, metrics, timeout, win_close, disp_size, disp_rows):
        batch_size = len(self.idxes)

        height = self.image_size + 20
        width = (self.image_size + 10) * batch_size + 10

        if self.canvas is None:
            self.canvas = np.full(shape=(height, width, 3), fill_value=255, dtype=np.uint8)
        else:
            self.canvas[:] = 255

        for n in range(batch_size):
            image_info = self.image_info_list[self.idxes[n]]
            filepath = self.image_path + image_info["file_name"]
        
            img = cv2.imread(filepath)
            
            height, width, _ = img.shape
            ratio = self.image_size / max(height, width)
            hratio = height / max(height, width)
            wratio = width / max(height, width)
            img_height = int(height * ratio)
            img_width = int(width * ratio)
            img = cv2.resize(img,(img_width, img_height))
            
            hbase = (self.image_size - img_height) // 2 + 10
            wbase = (self.image_size - img_width) // 2 + 10 + n * (self.image_size + 10)

            obj_confs = metrics["conf"][n]
            obj_classes = metrics["class"][n]
            obj_boxes = metrics["box"][n]

            for m in range(obj_confs.shape[0]):
                prob = obj_confs[m].item()
                if prob < 0.25: continue
                obj_class = obj_classes[m].item()
                box = obj_boxes[m]

                w = int(box[2].item() * wratio)
                h = int(box[3].item() * hratio)
                x = int(box[0].item() * wratio - w / 2)
                y = int(box[1].item() * hratio - h / 2)

                class_name = self.categories[int(obj_class)]
                color = (m//9*127, (m//3)%3*127, m%3*127)
                text = f"{class_name} {prob:.2f}"

                cv2.rectangle(img,(x,y),(x+w,y+h),color, 2)
                cv2.putText(img, text, (x+5, y+25), 5, 1, color);

            self.canvas[hbase:img_height+hbase, wbase:img_width+wbase,:] = img

        cv2.imshow("yolo_result", self.canvas)
        cv2.waitKey(timeout)
        if win_close: cv2.destroyAllWindows()
