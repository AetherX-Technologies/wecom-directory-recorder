# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .ocr_client import PaddleOCRVLClient
from .utils.logger import logger
from .utils.utils import read_img


class OCRProcessor:
    def __init__(
        self,
        ocr_params: Optional[Dict] = None,
        batch_size: int = 10,
        ocr_client: Optional[Any] = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")
        self.ocr_engine = ocr_client or self._init_ocr_engine(ocr_params)
        self.batch_size = batch_size

    def _init_ocr_engine(
        self, ocr_params: Optional[Dict] = None
    ) -> PaddleOCRVLClient:
        return PaddleOCRVLClient.from_env(ocr_params)

    def __call__(
        self, img_list: List[Path], is_batch_rec: bool, is_txt_dir: bool
    ) -> Tuple[List[str], List[str], List[str]]:
        self.is_txt_dir = is_txt_dir
        process_func = self.batch_rec if is_batch_rec else self.single_rec
        rec_results = process_func(img_list)
        srt_results = self._generate_srt_results(rec_results)
        ass_results = self._generate_ass_results(rec_results)
        txt_results = self._generate_txt_result(rec_results)
        return srt_results, ass_results, txt_results

    def single_rec(self, img_list: List[Path]) -> List[Tuple[int, str, str, str]]:
        logger.info("[OCR] Running with single recognition.")

        rec_results = []
        for i, img_path in enumerate(tqdm(img_list, desc="OCR")):
            rec_results.append(self._recognize_image(i, img_path))
        return rec_results

    @staticmethod
    def _get_srt_timestamp(file_path: Path) -> str:
        """0_00_00_041__0_00_00_415_0070000000019200080001920.jpeg"""

        def format_time(time_parts):
            time_parts[0] = f"{time_parts[0]:0>2}"
            return ":".join(time_parts[:3]) + f",{time_parts[3]}"

        split_paths = file_path.stem.split("_")
        start_time = split_paths[:4]
        end_time = split_paths[5:9]
        return f"{format_time(start_time)} --> {format_time(end_time)}"

    @staticmethod
    def _get_ass_timestamp(file_path: Path) -> str:
        s = file_path.stem

        h1 = int(s[0:1])
        m1 = int(s[2:4])
        sec1 = int(s[5:7])
        ms1 = int(s[8:11])

        h2 = int(s[13:14])
        m2 = int(s[15:17])
        sec2 = int(s[18:20])
        ms2 = int(s[21:24])

        # compute absolute times in milliseconds
        bt = (h1 * 3600 + m1 * 60 + sec1) * 1000 + ms1
        et = (h2 * 3600 + m2 * 60 + sec2) * 1000 + ms2

        def to_ass(ts_ms: int) -> str:
            # centiseconds (drop the last digit, no rounding)
            cs_total = ts_ms // 10
            cs = cs_total % 100
            total_s = ts_ms // 1000
            s = total_s % 60
            total_m = total_s // 60
            m = total_m % 60
            h = total_m // 60
            # H:MM:SS.CC
            return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

        return f"{to_ass(bt)},{to_ass(et)}"

    @staticmethod
    def _preprocess_image(img_path: Path) -> np.ndarray:
        img = read_img(img_path)
        if img is None:
            raise ValueError("Unable to read OCR image: {}".format(img_path))
        return img

    @staticmethod
    def _generate_srt_results(
        rec_results: List[Tuple[int, str, str, str]],
    ) -> List[str]:
        return [f"{i + 1}\n{time_str}\n{txt}\n" for i, time_str, txt, _ in rec_results]

    @staticmethod
    def _generate_ass_results(
        rec_results: List[Tuple[int, str, str, str]],
    ) -> List[str]:
        return [
            f"Dialogue: 0,{ass_time_str},Default,,0,0,0,,{txt}"
            for _, _, txt, ass_time_str in rec_results
        ]

    @staticmethod
    def _generate_txt_result(rec_results: List[Tuple[int, str, str, str]]) -> List[str]:
        return [f"{txt}\n" for _, _, txt, _ in rec_results]

    def batch_rec(self, img_list: List[Path]) -> List[Tuple[int, str, str, str]]:
        logger.info("[OCR] Running with batched remote recognition.")

        img_nums = len(img_list)
        rec_results = []
        for start_i in tqdm(range(0, img_nums, self.batch_size), desc="OCR batches"):
            end_i = min(img_nums, start_i + self.batch_size)
            for image_index in range(start_i, end_i):
                rec_results.append(
                    self._recognize_image(image_index, img_list[image_index])
                )
        return rec_results

    def _recognize_image(
        self, image_index: int, img_path: Path
    ) -> Tuple[int, str, str, str]:
        image = self._preprocess_image(img_path)
        text = self.get_ocr_result(image)
        return (
            image_index,
            self._get_srt_timestamp(img_path),
            text,
            self._get_ass_timestamp(img_path),
        )

    def get_ocr_result(self, img: np.ndarray) -> str:
        return self.ocr_engine.recognize(img)
