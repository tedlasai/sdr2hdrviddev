from networkx import center
import torch, torchvision, imageio, os, json, pandas
import imageio.v3 as iio
from PIL import Image
import os
    
from pathlib import Path
import re
import cv2
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)



class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)



class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data



class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)



class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)



class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)



class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True):
        self.convert_RGB = convert_RGB
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        return image



class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor, crop_size_h=None, crop_size_w=None, crf_aug=None):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.crop_size_h = crop_size_h
        self.crop_size_w = crop_size_w
        self.crf_aug = crf_aug

    def crop_and_resize(self, image, target_height, target_width):
        if isinstance(image, Image.Image):
            width, height = image.size
            scale = max(target_width / width, target_height / height)
            image = torchvision.transforms.functional.resize(
                image,
                (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )
            image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))

            #randomly sample a crop of size crop_size (if image is not big enough throw error)
            if self.crop_size_h is not None and self.crop_size_w is not None:
                assert target_height >= self.crop_size_h and target_width >= self.crop_size_w, f"Image size {target_height}x{target_width} is smaller than crop size {self.crop_size_h}x{self.crop_size_w}"
                top = np.random.randint(0, target_height - self.crop_size_h + 1)
                left = np.random.randint(0, target_width - self.crop_size_w + 1)
                image = torchvision.transforms.functional.crop(image, top, left, self.crop_size_h, self.crop_size_w)

        elif isinstance(image, np.ndarray):
            height, width = image.shape[:2]
            scale = max(target_width / width, target_height / height)
            new_size = (round(height * scale), round(width * scale))

            # to tensor (C,H,W)
            tensor_img = torch.from_numpy(image.transpose(2, 0, 1)).float()
            resized = torchvision.transforms.functional.resize(
                tensor_img,
                new_size,
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            )

            # center crop
            cropped = torchvision.transforms.functional.center_crop(
                resized, (target_height, target_width)
            )


            if self.crop_size_h is not None and self.crop_size_w is not None:
                assert target_height >= self.crop_size_h and target_width >= self.crop_size_w, f"Image size {target_height}x{target_width} is smaller than crop size {self.crop_size_h}x{self.crop_size_w}"
                top = np.random.randint(0, target_height - self.crop_size_h + 1)
                left = np.random.randint(0, target_width - self.crop_size_w + 1)
                cropped = torchvision.transforms.functional.crop(cropped, top, left, self.crop_size_h, self.crop_size_w)

            # back to numpy (H,W,C)
            image = cropped.permute(1, 2, 0).numpy()


        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def __call__(self, data):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image



class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    




def _natural_key(s: str):
    # splits into digit/non-digit chunks so "file2" < "file10"
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]

def next_paths(path: str, t: int, *, same_suffix: bool = True, include_self: bool = False):
    """
    Given a file path, return the next t paths in that directory in natural sorted order.
    
    Args:
        path: The starting file path.
        t: How many following paths to return.
        same_suffix: If True, only consider files with the same extension as `path`.
        include_self: If True, include `path` itself as the first element (then next t-1).
        
    Returns:
        List[str]: up to t subsequent paths (or t including self if include_self=True).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {path}")

    directory = p.parent
    suffix = p.suffix

    # Gather candidates
    if same_suffix:
        candidates = [f for f in directory.iterdir() if f.is_file() and f.suffix == suffix]
    else:
        candidates = [f for f in directory.iterdir() if f.is_file()]

    # Sort naturally by name
    candidates.sort(key=lambda f: _natural_key(f.name))

    # Find index of the given file
    try:
        idx = candidates.index(p)
    except ValueError:
        # If the file isn't in the filtered list (e.g., suffix mismatch), fall back to all files
        all_files = sorted([f for f in directory.iterdir() if f.is_file()],
                           key=lambda f: _natural_key(f.name))
        try:
            idx = all_files.index(p)
            candidates = all_files  # use this list going forward
        except ValueError:
            raise FileNotFoundError(f"File not found among directory listings: {path}")

    # Slice out next t entries (optionally including self)
    start = idx if include_self else idx + 1
    result = candidates[start:start + t]

    return [str(f) for f in result]

def exposure_scale(frame, p, mode, lo=0.0, hi=1.0, eps=1e-8):

    if mode == "over":   # p pixels clip to hi
        #do max over channels
        x = frame.max(axis=2).ravel()
        q = np.quantile(x, 1.0 - p)
        return hi / (q + eps)
    if mode == "under":  # p pixels fall below 0.5/255 after gamma
        q = -10
        while q < 0:
            #do min over channels
            x = frame.min(axis=2).ravel()
            q = np.quantile(x, p)
            p+=0.05
        return ((0.5/ 255) ** 2.2)  / (q + eps)
    raise ValueError("mode must be 'over' or 'under'")


def make_exposure_brackets(hdr_paths, frame_processor, exposures=[0,-4, 4], crf_aug=None):
    """
    Given a list of HDR image paths, generate exposure-bracketed LDR images.

    Args:
        hdr_paths (list[str]): Paths to HDR images.
        exposures (tuple[int|float]): EV values to apply for exposure scaling.

    Returns:
        list[list[np.ndarray]]: For each HDR path, a list of LDR images
                                (same order as exposures).
    """

    if crf_aug == "random":
        #modes = ["crf", "nocrf"] #for 10 epochs

        modes = ["crf"]#,"crf_extend"]#  "nocrf","nocrf_extend"] #for last 10 epochs
        input_type = np.random.choice(modes)
    else:
        input_type = "crf"
    
    # if "extend" not in input_type:
    #     hdr_paths = hdr_paths[:-4] #remove the last 4 hdr_paths


    all_brackets = []
    hdr_images = []
    ldr_w_crf_images = []
    prev_noise = None
 
    for i, hdr_path in enumerate(hdr_paths):
        hdr_in = cv2.imread( hdr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[:, :, ::-1]

        data_type = "stuttgart"
        if data_type == "stuttgart":
            hdr_in = hdr_in[10:-10, 10:-10, :]  #remove 10 pixel black border



        max_value = np.max(hdr_in)
        median_value = np.percentile(hdr_in, 50)
        min_exposure = np.log2(exposure_scale(hdr_in, 0.1, "under")) #np.log(1/ max_value)
        max_exposure = np.log2(exposure_scale(hdr_in, 0.3, "over")) #np.log2(1/median_value)
        ae_exposure = np.log2(0.18 / hdr_in.mean())

        min_in_exposure = np.log2(0.05 / hdr_in.mean())
        max_in_exposure = np.log2(0.7 / hdr_in.mean())

        if i == 0:
            max_offset = 4
            if crf_aug == "random":
                predict_mode = "both"
                if (min_exposure) < (max_exposure):
                    center = np.random.uniform(min_exposure, max_exposure)
                else:
                    center = (min_exposure + max_exposure)//2 #this shouldn't be reached that often
            else:
                center = max_in_exposure
                #center = 0

            # else:
            #     center = max_in_exposure #use max_over exposure for center
            #     neg_room = 4
            #     pos_room = 4


            # now pick offsets that still fit
            neg = -4
            pos =  4

            
            exposures[1] = neg
            exposures[2] = pos
            
            random_scale = center

        hdr_in = hdr_in * 2**(random_scale)

            #cache[hdr_path] = hdr_in
        hdr_in =np.clip(hdr_in, 0.0, 2**max(exposures)) #CLIP
        hdr_in = frame_processor(hdr_in)

        hdr_images.append(hdr_in)

        ldr_images = []
        for ev in exposures:
            # Scale exposure (2^EV), clip to [0,1]
            ldr = np.clip(hdr_in * (2.0 ** ev), 0.0, 1.0)
            ldr = (ldr * 255.0)
            ldr_images.append(ldr)
        all_brackets.append(ldr_images)

        if i== 0:
            if crf_aug == "random":
                # Randomly gamma
                n = np.random.normal(0.9, 0.1)
                sigma = np.random.normal(0.6, 0.1)
                # enforce sane ranges
                n = max(n, 0.1)
                sigma = max(sigma, 0.1)

                sigma_s = np.random.uniform(0.0, 0.05)   # shot noise coeff (0-1 scale); max std ~0.14 at full white
                sigma_r = np.random.uniform(0.0, 0.02)  # read noise std (0-1 scale); max std 0.05
            else:
                n = 0.9
                sigma = 0.6
                
        mid_exposure = exposures[0]
        radiance = np.clip((hdr_in * (2.0 ** mid_exposure)), 0.0, 1.0)

        if crf_aug == "random":
            noise_std = np.sqrt((sigma_s**2) * radiance + (sigma_r ** 2))
            u_t = np.random.normal(0.0, 1.0, radiance.shape)
            if prev_noise is None:
                epsilon_t = u_t
            else:
                rho = 0.5
                epsilon_t = rho * prev_noise + np.sqrt(1 - rho ** 2) * u_t
            prev_noise = epsilon_t
            radiance = radiance + epsilon_t * noise_std
            radiance = np.clip(radiance, 0.0, 1.0)

        Hn = np.power(radiance, n)
        ldr_w_crf = (1 + sigma) * Hn / (Hn + sigma)

        ldr_w_crf = np.clip(ldr_w_crf, 0.0, 1.0)
        #print if any value is above 1 or below 0
        if not np.all(ldr_w_crf >= 0.0) or not np.all(ldr_w_crf <= 1.0):
            print(f"Warning: CRF applied LDR has values outside [0,1] for image {hdr_path}")
        assert np.all(ldr_w_crf >= 0.0) and np.all(ldr_w_crf <= 1.0), "LDR with CRF has values outside [0,1]"
        ldr_w_crf = (ldr_w_crf * 255.0).round().astype(np.uint8)
        ldr_w_crf = ldr_w_crf.astype(np.float32) #quantize and back to float32 

        #do inverse of CRF to get back to linear
        # ldr_w_crf = ldr_w_crf.astype(np.float32) / 255.0
        # ldr_w_crf = (sigma * ldr_w_crf) / ((1 + sigma) - ldr_w_crf)
        # ldr_w_crf = np.power(np.clip(ldr_w_crf, 0.0, 1.0), 1.0 / n)
        # ldr_w_crf = ldr_w_crf * (255.0)

        ldr_w_crf_images.append(ldr_w_crf)


    hdr_images = np.array(hdr_images)  # shape (N, H, W, 3)
    ldr_w_crf_images = np.array(ldr_w_crf_images)  # shape (N, H, W, 3)
    all_brackets = np.array(all_brackets)  # shape (N, len(exposures), H, W, 3)
    all_brackets = all_brackets.transpose(1,0,2,3,4)  # shape (len(exposures), N, H, W, 3)

    #add crf_video as first bracket
    all_brackets = np.concatenate([ldr_w_crf_images[None, ...], all_brackets], axis=0)  # shape (len(exposures)+1, N, H, W, 3)

    exposures = np.array(exposures) 
    exposures = np.concatenate(([0], exposures))/4  # add 0 for crf_video

    #make input_type == "crf" 50% of time and "nocrf" 50% of time
    #input_type = "crf" if np.random.rand() < 0.5 else "nocrf"
    #modes = 
    


    #extend versions are wrong, but this will work for now...
    if input_type in ["crf", "crf_extend"]:
        input_video = all_brackets[0]  # use the crf_video as input
    elif input_type in ["nocrf", "nocrf_extend"]:
        input_video = all_brackets[1]  # use the first exposure bracket as input
        
    data = {"hdr_video": hdr_images, "bracket_video": all_brackets, "input_video": input_video, "exposures": exposures, "input_type": input_type}
    return data

class LoadHDRVideo(DataProcessingOperator):
    def __init__(self, num_frames=49, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x, crf_aug=None):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        self.crf_aug = crf_aug
        self.cache = {}

    # def get_num_frames(self, reader):
    #     num_frames = self.num_frames
    #     if int(reader.count_frames()) < num_frames:
    #         num_frames = int(reader.count_frames())
    #         while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
    #             num_frames -= 1
    #     return num_frames
        
    def __call__(self, data: str):
        #get num_frames-1 next frames with same suffix 
        #assert that num_frames is 3n
        assert (self.num_frames) % 3 == 0, "num_frames must be 3n"
        num_hdr_frames = (self.num_frames) // 3
        import random
        if self.crf_aug == "random":
            num_hdr_frames = 17#7 # random.choice(num_frames_train)  #try 17 (5 latent frames per)

        else:
            num_hdr_frames = 17#7

        #num_hdr_frames += 4 #handle extend cases

        hdr_paths = next_paths(data, num_hdr_frames, same_suffix=True, include_self=True)

        data = make_exposure_brackets(hdr_paths, self.frame_processor, crf_aug=self.crf_aug)
        data["bracket_video"] = data["bracket_video"].reshape(-1, *data["bracket_video"].shape[2:])  # shape (num_frames, H, W, 3)


        return data


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]




class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")



class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")



class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)



class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)



class StuttgartDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None,
        repeat=1,
        main_data_operator=lambda x: x,
        special_operator_map=None,
        mode = "brackets",
        split = "train",
    ):
        self.base_path = base_path
        self.repeat = repeat
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.mode = mode
        self.data = []
        self.cached_data = {}
        self.load_from_cache = False
        self.split = split
        self.load_data_from_path()
    
            
    def load_data_from_path(self):
        self.data = []

        #only val
        only_val = ["bistro_01", "bistro_02", "bistro_03", "showgirl_01", "showgirl_02", "smith_welding", "carousel_fireworks_02", "fireplace_01", "hdr_testimage"]
        for root, dirs, files in os.walk(self.base_path):
            if files == []:
                continue  # skip folders that only contain subfolders
            files = sorted(files)[:-17]
            
            if not files:
                continue

            if self.split == "val":
                if any(val_name in root for val_name in only_val):
                    files = files[:1]
                else:
                    continue
            elif self.split == "train":
                if any(val_name in root for val_name in only_val):
                    continue  # skip validation videos during training

            # if "fireworks_02" not in root:
            #     continue  # TEMPORARY: only use fireworks_02 for testing

            for f in files:
                self.data.append({
                    "video": os.path.relpath(os.path.join(root, f), self.base_path),
                })

        if self.split == "val":
            self.data = self.data[:]


    @staticmethod
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
        crop_size_h=None, crop_size_w=None, 
        crf_aug=None,
    ):
        return RouteByType(operator_map=[(str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("hdr", "exr"), LoadHDRVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor, crop_size_h=crop_size_h, crop_size_w=crop_size_w),
                    crf_aug=crf_aug,
                )),
            ]))
        ])

            

    def __getitem__(self, data_id):
        #choose random data_id between 0 and len(self.data)
        if self.split == "train":
            data_id = np.random.randint(0, len(self.data))
        if (data_id % len(self.data)) in self.cached_data:
            data = self.cached_data[data_id % len(self.data)]
        else:
            data = self.data[data_id % len(self.data)].copy()
            if self.mode == "hdr_and_brackets":
                data = self.main_data_operator(data["video"])

            data["prompt"] = ""
            #self.cached_data[data_id % len(self.data)] = data
        return data

    def __len__(self):

        if self.split == "val":
            return min(5, len(self.data))  # Use only last 20 samples for validation
        if self.split == "train":
            return int(len(self.data)/4)
        else:
            return len(self.data)

        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
    
