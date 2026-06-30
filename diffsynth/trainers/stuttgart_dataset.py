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

DATASETS = ["stuttgart"]  # e.g. ["stuttgart", "invert_pipeline", "rawhdr", "hdrps_raws"]
INVERT_PIPELINE_PATH = "/data2/saikiran.tedla/hdrvideo/diff/data/invert_pipeline"
RAWHDR_PATH = "/data2/saikiran.tedla/hdrvideo/diff/data/RawHDR"
HDRPS_RAWS_PATH = "/data2/saikiran.tedla/hdrvideo/diff/data/HDRPS_Raws"
# EXR cache: RawHDR/RawHDRTrain_EXR, RawHDR/RawHDRTest_EXR — run: python -m diffsynth.trainers.rawhdr_dataset

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
        x = frame.min(axis=2).ravel()
        while q < 0 and p <= 1.0:
            q = np.quantile(x, min(p, 1.0))
            p += 0.05
        if q <= 0:
            q = eps  # all pixels are zero/negative; avoid div-by-zero
        return ((0.5/ 255) ** 2.2)  / (q + eps)
    raise ValueError("mode must be 'over' or 'under'")


def make_exposure_brackets(hdr_paths, frame_processor, exposures=[0,-4, 4], crf_aug=None, predict_mode="default", include_crf_in_brackets=True, exp_gap=7, bracket_mode="flex_brackets", predict_gamma=True, fixed_exposures=None):
    """
    Given a list of HDR image paths, generate exposure-bracketed LDR images.

    Args:
        hdr_paths (list[str]): Paths to HDR images.
        exposures (tuple[int|float]): EV values to apply for exposure scaling.
        bracket_mode (str): "flex_brackets" scales HDR so the sequence global max
            maps to the darkest bracket peak (MAP_MAX); "fixed_brackets" uses
            (-exp_gap, 0, +exp_gap) EV relative to crf_radiance without that scaling;
            "fixed_brackets_3" randomly trains with (-4,0,4), (0,4,8), or (-8,-4,0).
        fixed_exposures (tuple|None): when set, overrides bracket offsets explicitly
            (used for per-mode validation with fixed_brackets_3).

    Returns:
        list[list[np.ndarray]]: For each HDR path, a list of LDR images
                                (same order as exposures).
    """
    if bracket_mode not in ("flex_brackets", "fixed_brackets", "fixed_brackets_3"):
        raise ValueError(f"bracket_mode must be 'flex_brackets', 'fixed_brackets', or 'fixed_brackets_3', got {bracket_mode!r}")

    input_type = "crf"

    # --- Pass 1: load all raw frames (border crop only) ---
    raw_frames = []
    for hdr_path in hdr_paths:
        hdr_in = cv2.imread(hdr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[:, :, ::-1].copy()
        hdr_in = hdr_in[10:-10, 10:-10, :]  # remove 10 pixel black border
        raw_frames.append(hdr_in)

    if fixed_exposures is not None:
        exposures = list(fixed_exposures)
    elif bracket_mode == "fixed_brackets_3":
        g = int(exp_gap)
        _FIXED_3_SETS = [(-g, 0, g), (0, g, 2*g), (-2*g, -g, 0)]
        exposures = list(_FIXED_3_SETS[np.random.randint(len(_FIXED_3_SETS))])
    else:
        g = int(exp_gap)
        exposures = [-g, 0, g]

    if bracket_mode == "flex_brackets":
        global_max = max(np.percentile(frame_processor(f), 99.9) for f in raw_frames)
        MAP_MAX = 0.9
        fit_scale = MAP_MAX / (global_max * 2**exposures[0])  # robust 99.9th percentile → darkest bracket peaks at MAP_MAX
    else:
        fit_scale = 1.0

    if crf_aug == "random":
        n = np.random.normal(0.9, 0.1)
        sigma = np.random.normal(0.6, 0.1)
        n = max(n, 0.1)
        sigma = max(sigma, 0.1)
        sigma_s = np.random.uniform(0.0, 0.05)
        sigma_r = np.random.uniform(0.0, 0.02)
    else:
        n = 0.9
        sigma = 0.6

    # --- Pass 2: apply frame_processor, compute brackets and CRF ---
    all_brackets = []
    hdr_images = []
    ldr_w_crf_images = []
    prev_noise = None

    for i, hdr_in in enumerate(raw_frames):
        hdr_in = frame_processor(hdr_in)
        hdr_in = hdr_in * fit_scale
        hdr_images.append(hdr_in)

        # --- Compute sequence-level parameters from frame 0 ---
        if i == 0:
            min_exposure = np.log2(exposure_scale(hdr_in, 0.3, "under"))
            max_exposure = np.log2(exposure_scale(hdr_in, 0.3, "over"))
            max_in_exposure = np.log2(0.7 / max(hdr_in.mean(), 1e-8))

            if crf_aug == "random":
                if min_exposure < max_exposure:
                    center = np.random.uniform(min_exposure, max_exposure)
                else:
                    center = (min_exposure + max_exposure) / 2.0
            else:
                center = max_in_exposure

        crf_radiance = hdr_in * 2**center

        radiance_ref = crf_radiance if bracket_mode in ("fixed_brackets", "fixed_brackets_3") else hdr_in
        ldr_images = []
        for ev in exposures:
            ldr = np.clip(radiance_ref * (2.0 ** ev), 0.0, 1.0)
            if predict_gamma:
                ldr = ldr ** (1/2.2)
            ldr = (ldr * 255.0)
            ldr_images.append(ldr)
        all_brackets.append(ldr_images)

        crf_radiance_clipped = np.clip(crf_radiance, 0.0, 1.0)
        if crf_aug == "random":
            noise_std = np.sqrt((sigma_s**2) * crf_radiance_clipped + (sigma_r ** 2))
            u_t = np.random.normal(0.0, 1.0, crf_radiance_clipped.shape)
            if prev_noise is None:
                epsilon_t = u_t
            else:
                rho = 0.5
                epsilon_t = rho * prev_noise + np.sqrt(1 - rho ** 2) * u_t
            prev_noise = epsilon_t
            crf_radiance_clipped = crf_radiance_clipped + epsilon_t * noise_std
            crf_radiance_clipped = np.clip(crf_radiance_clipped, 0.0, 1.0)

        Hn = np.power(crf_radiance_clipped, n)
        ldr_w_crf = (1 + sigma) * Hn / (Hn + sigma)

        ldr_w_crf = np.clip(ldr_w_crf, 0.0, 1.0)
        if not np.all(ldr_w_crf >= 0.0) or not np.all(ldr_w_crf <= 1.0):
            print(f"Warning: CRF applied LDR has values outside [0,1] for frame {i}")
        assert np.all(ldr_w_crf >= 0.0) and np.all(ldr_w_crf <= 1.0), "LDR with CRF has values outside [0,1]"
        ldr_w_crf = (ldr_w_crf * 255.0).round().astype(np.uint8)
        ldr_w_crf = ldr_w_crf.astype(np.float32) #quantize and back to float32 
        ldr_w_crf_images.append(ldr_w_crf)


    hdr_images = np.array(hdr_images)  # shape (N, H, W, 3)
    ldr_w_crf_images = np.array(ldr_w_crf_images)  # shape (N, H, W, 3)
    all_brackets = np.array(all_brackets)  # shape (N, len(exposures), H, W, 3)
    all_brackets = all_brackets.transpose(1,0,2,3,4)  # shape (len(exposures), N, H, W, 3)

    exposures = np.array(exposures)
    if include_crf_in_brackets:
        all_brackets = np.concatenate([ldr_w_crf_images[None, ...], all_brackets], axis=0)  # shape (len(exposures)+1, N, H, W, 3)
        exposures = np.concatenate(([0], exposures))

    if input_type in ["crf", "crf_extend"]:
        input_video = ldr_w_crf_images if not include_crf_in_brackets else all_brackets[0]
    elif input_type in ["nocrf", "nocrf_extend"]:
        input_video = all_brackets[0]

    data = {
        "hdr_video": hdr_images,
        "bracket_video": all_brackets,
        "input_video": input_video,
        "exposures": exposures,
        "input_type": input_type,
        "include_crf_in_brackets": include_crf_in_brackets,
        "bracket_mode": bracket_mode,
    }
    return data

class LoadHDRVideo(DataProcessingOperator):
    def __init__(self, num_frames=49, num_hdr_frames=17, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x, crf_aug=None, predict_mode="default", include_crf_in_brackets=True, exp_gap=7, bracket_mode="flex_brackets", predict_gamma=True, fixed_exposures=None):
        self.num_frames = num_frames
        self.num_hdr_frames = num_hdr_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        self.crf_aug = crf_aug
        self.predict_mode = predict_mode
        self.include_crf_in_brackets = include_crf_in_brackets
        self.exp_gap = int(exp_gap)
        self.bracket_mode = bracket_mode
        self.predict_gamma = predict_gamma
        self.fixed_exposures = fixed_exposures
        self.cache = {}

    # def get_num_frames(self, reader):
    #     num_frames = self.num_frames
    #     if int(reader.count_frames()) < num_frames:
    #         num_frames = int(reader.count_frames())
    #         while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
    #             num_frames -= 1
    #     return num_frames
        
    def __call__(self, data: str):
        num_hdr_frames = self.num_hdr_frames

        # Some datasets are actually single-image sources (e.g. RawHDR stored as
        # individual EXRs). For those, repeat the same image across the requested
        # HDR-frame count.
        if Path(data).parent.name in ["RawHDRTrain_EXR", "RawHDRTest_EXR"]:
            hdr_paths = [data] * num_hdr_frames
        else:
            hdr_paths = next_paths(data, num_hdr_frames, same_suffix=True, include_self=True)
            if len(hdr_paths) < num_hdr_frames:
                hdr_paths = [hdr_paths[0]] * num_hdr_frames

        data = make_exposure_brackets(
            hdr_paths,
            self.frame_processor,
            crf_aug=self.crf_aug,
            predict_mode=self.predict_mode,
            include_crf_in_brackets=self.include_crf_in_brackets,
            exp_gap=self.exp_gap,
            bracket_mode=self.bracket_mode,
            predict_gamma=self.predict_gamma,
            fixed_exposures=self.fixed_exposures,
        )
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
        datasets=None,
        invert_pipeline_path=None,
        rawhdr_path=None,
        hdrps_raws_path=None,
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
        self.datasets = DATASETS if datasets is None else datasets
        self.invert_pipeline_path = INVERT_PIPELINE_PATH if invert_pipeline_path is None else invert_pipeline_path
        self.rawhdr_path = RAWHDR_PATH if rawhdr_path is None else rawhdr_path
        self.hdrps_raws_path = HDRPS_RAWS_PATH if hdrps_raws_path is None else hdrps_raws_path

        self.OVERFITTING = False
        self.load_data_from_path()
    
            
    def load_data_from_path(self):
        self.data = []
        if "stuttgart" in self.datasets:
            self._load_stuttgart_data()
        if "invert_pipeline" in self.datasets:
            self._load_invert_pipeline_data()
        if "rawhdr" in self.datasets:
            self._load_rawhdr_data()
        if "hdrps_raws" in self.datasets:
            self._load_hdrps_raws_data()

        if self.OVERFITTING:
            self.data = [{"video": "/data2/saikiran.tedla/hdrvideo/diff/data/stuttgart/carousel_fireworks_02/carousel_fireworks_02_000936.exr"}]

    def _load_stuttgart_data(self):
        if not self.base_path:
            raise ValueError("base_path is required when using the stuttgart dataset")

        only_val = ["bistro_01", "bistro_02", "bistro_03", "showgirl_01", "showgirl_02", "smith_welding", "carousel_fireworks_02", "fireplace_01", "hdr_testimage"]
        for root, dirs, files in os.walk(self.base_path):
            if files == []:
                continue
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
                    continue

            for f in files:
                self.data.append({
                    "video": os.path.relpath(os.path.join(root, f), self.base_path),
                    "dataset": "stuttgart",
                })

    def _load_invert_pipeline_data(self):
        if self.split != "train":
            return

        root = Path(self.invert_pipeline_path)
        if not root.is_dir():
            raise FileNotFoundError(f"invert_pipeline path does not exist: {root}")

        gt_hdr_paths = sorted(root.rglob("gt.hdr"), key=lambda p: str(p))
        for gt_hdr in gt_hdr_paths:
            self.data.append({
                "video": str(gt_hdr.resolve()),
                "dataset": "invert_pipeline",
            })

    def _load_rawhdr_data(self):
        if self.split != "train":
            return
        # We treat RawHDR as an image dataset:
        # - train split reads RawHDRTrain_EXR
        # - val split reads RawHDRTest_EXR
        root = Path(self.rawhdr_path)
        if not root.is_dir():
            raise FileNotFoundError(f"rawhdr path does not exist: {root}")

        exr_root = root / "RawHDRTrain_EXR"

        if not exr_root.is_dir():
            raise FileNotFoundError(f"RawHDR EXR directory does not exist: {exr_root}")

        exr_paths = sorted(exr_root.glob("*.exr"), key=lambda p: str(p))
        for exr_path in exr_paths:
            self.data.append({
                "video": str(exr_path.resolve()),
                "dataset": "rawhdr",
            })

    def _load_hdrps_raws_data(self):
        if self.split != "train":
            return

        root = Path(self.hdrps_raws_path)
        if not root.is_dir():
            raise FileNotFoundError(f"hdrps_raws path does not exist: {root}")

        exr_paths = sorted(root.rglob("*.exr"), key=lambda p: str(p))
        for exr_path in exr_paths:
            self.data.append({
                "video": str(exr_path.resolve()),
                "dataset": "hdrps_raws",
            })

    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, num_hdr_frames=17, time_division_factor=4, time_division_remainder=1,
        crop_size_h=None, crop_size_w=None,
        crf_aug=None,
        predict_mode="default",
        include_crf_in_brackets=True,
        exp_gap=7,
        bracket_mode="flex_brackets",
        predict_gamma=True,
        fixed_exposures=None,
    ):
        return RouteByType(operator_map=[(str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("hdr", "exr"), LoadHDRVideo(
                    num_frames, num_hdr_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor, crop_size_h=crop_size_h, crop_size_w=crop_size_w),
                    crf_aug=crf_aug,
                    predict_mode=predict_mode,
                    include_crf_in_brackets=include_crf_in_brackets,
                    exp_gap=exp_gap,
                    bracket_mode=bracket_mode,
                    predict_gamma=predict_gamma,
                    fixed_exposures=fixed_exposures,
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
        if self.OVERFITTING:
            if self.split == "val":
                return 1
            elif self.split == "train":
                return 500
            else:
                return 1

        if self.split == "val":
            return min(5, len(self.data))

        n = len(self.cached_data) if self.load_from_cache else len(self.data)
        if self.split == "train":
            return int(n * self.repeat)
            return int(n * self.repeat)
        return n
        
    def set_predict_mode(self, mode):
        op = self.main_data_operator
        for _, pipeline in op.operator_map:
            ops = pipeline.operators if isinstance(pipeline, DataProcessingPipeline) else [pipeline]
            for o in ops:
                if isinstance(o, RouteByExtensionName):
                    for _, loader in o.operator_map:
                        if isinstance(loader, LoadHDRVideo):
                            loader.predict_mode = mode
                            return

    def set_resolution(self, height, width):
        self.cached_data = {}
        op = self.main_data_operator
        for _, pipeline in op.operator_map:
            ops = pipeline.operators if isinstance(pipeline, DataProcessingPipeline) else [pipeline]
            for o in ops:
                if isinstance(o, RouteByExtensionName):
                    for _, loader in o.operator_map:
                        if isinstance(loader, LoadHDRVideo):
                            loader.frame_processor.height = height
                            loader.frame_processor.width = width
                            return
        raise RuntimeError("Could not find ImageCropAndResize in main_data_operator")

    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
    
