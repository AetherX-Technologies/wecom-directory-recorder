# Public integration modification: synthetic fixtures or removal of private defaults; see THIRD_PARTY_NOTICES.md.
# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import shutil
import sys
from pathlib import Path

import pytest
import cv2
import numpy as np

cur_dir = Path(__file__).resolve().parent
root_dir = cur_dir.parent

sys.path.append(str(root_dir))

from rapid_videocr import RapidVideOCR, RapidVideOCRExeception, RapidVideOCRInput
from rapid_videocr.utils.utils import mkdir, read_txt

test_dir = cur_dir / "test_files"


@pytest.fixture(scope="module", autouse=True)
def synthetic_inputs(tmp_path_factory):
    global test_dir
    test_dir = tmp_path_factory.mktemp("subtitle-fixtures")
    timings = [("0_00_00_041", "0_00_00_415"), ("0_00_00_416", "0_00_01_165"),
               ("0_00_01_166", "0_00_01_540"), ("0_00_01_541", "0_00_02_540")]
    for folder in ("RGBImages", "TXTImages"):
        destination = test_dir / folder
        destination.mkdir()
        for index, (start, end) in enumerate(timings):
            image = np.full((80, 1920, 3), 255, dtype=np.uint8)
            cv2.putText(image, f"SYNTHETIC {index}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            ok, encoded = cv2.imencode('.jpeg', image)
            assert ok
            encoded.tofile(destination / f"{start}__{end}_0070000000019200080001920.jpeg")


class FakeOCRClient:
    def __init__(self):
        self.results = iter(
            [
                "空间里面他绝对赢不了的",
                "测试字幕二",
                "测试字幕三",
                "你们接着善后",
            ]
        )

    def recognize(self, image):
        return next(self.results)


def make_input(**kwargs):
    return RapidVideOCRInput(ocr_client=FakeOCRClient(), **kwargs)


@pytest.fixture
def setup_and_teardown():
    save_dir = test_dir / "tmp"
    mkdir(save_dir)

    srt_path = save_dir / "result.srt"
    ass_path = save_dir / "result.ass"
    txt_path = save_dir / "result.txt"

    yield save_dir, srt_path, ass_path, txt_path

    shutil.rmtree(save_dir)


@pytest.mark.parametrize(
    "img_dir",
    [test_dir / "RGBImages", test_dir / "TXTImages"],
)
def test_single_rec(setup_and_teardown, img_dir):
    img_dir = test_dir / img_dir.name
    save_dir, srt_path, ass_path, txt_path = setup_and_teardown

    extractor = RapidVideOCR(make_input())
    extractor(img_dir, save_dir)

    srt_data = read_txt(srt_path)
    assert len(srt_data) == 16
    assert srt_data[2] == "空间里面他绝对赢不了的"
    assert srt_data[-2] == "你们接着善后"

    ass_data = read_txt(ass_path)
    assert len(ass_data) == 17
    assert ass_data[13].split(",", 9)[-1] == "空间里面他绝对赢不了的"
    assert ass_data[-1].split(",", 9)[-1] == "你们接着善后"

    txt_data = read_txt(txt_path)
    assert len(txt_data) == 8
    assert txt_data[-2] == "你们接着善后"


@pytest.mark.parametrize("img_dir", [test_dir / "RGBImages"])
def test_concat_rec(setup_and_teardown, img_dir):
    img_dir = test_dir / img_dir.name
    save_dir, srt_path, ass_path, txt_path = setup_and_teardown

    input_param = make_input(is_batch_rec=True)
    extractor = RapidVideOCR(input_param)
    extractor(img_dir, save_dir)

    srt_data = read_txt(srt_path)
    assert len(srt_data) == 16
    assert srt_data[2] == "空间里面他绝对赢不了的"
    assert srt_data[-2] == "你们接着善后"

    ass_data = read_txt(ass_path)
    assert len(ass_data) == 17
    assert ass_data[13].split(",", 9)[-1] == "空间里面他绝对赢不了的"
    assert ass_data[-1].split(",", 9)[-1] == "你们接着善后"

    txt_data = read_txt(txt_path)
    assert len(txt_data) == 8
    assert txt_data[-2] == "你们接着善后"


@pytest.mark.parametrize(
    "img_dir",
    [test_dir / "RGBImage", test_dir / "TXTImage"],
)
def test_empty_dir(img_dir):
    img_dir = test_dir / img_dir.name
    extractor = RapidVideOCR(make_input())
    mkdir(img_dir)

    with pytest.raises(RapidVideOCRExeception) as exc_info:
        extractor(img_dir, test_dir)
    assert exc_info.type is RapidVideOCRExeception

    shutil.rmtree(img_dir)


@pytest.mark.parametrize(
    "img_dir",
    [test_dir / "RGBImage", test_dir / "TXTImage"],
)
def test_nothing_dir(img_dir):
    img_dir = test_dir / img_dir.name
    extractor = RapidVideOCR(make_input())
    mkdir(img_dir)
    with pytest.raises(RapidVideOCRExeception) as exc_info:
        extractor(img_dir, test_dir)
    assert exc_info.type is RapidVideOCRExeception

    shutil.rmtree(img_dir)


def test_out_only_srt(setup_and_teardown):
    save_dir, srt_path, ass_path, txt_path = setup_and_teardown

    img_dir = test_dir / "RGBImages"
    input_param = make_input(is_batch_rec=True, out_format="srt")
    extractor = RapidVideOCR(input_param)
    extractor(img_dir, save_dir)

    srt_data = read_txt(srt_path)
    assert len(srt_data) == 16
    assert srt_data[2] == "空间里面他绝对赢不了的"
    assert srt_data[-2] == "你们接着善后"


def test_out_only_ass(setup_and_teardown):
    save_dir, srt_path, ass_path, txt_path = setup_and_teardown

    img_dir = test_dir / "RGBImages"
    input_param = make_input(is_batch_rec=True, out_format="ass")
    extractor = RapidVideOCR(input_param)
    extractor(img_dir, save_dir)

    ass_data = read_txt(ass_path)
    assert len(ass_data) == 17
    assert ass_data[13].split(",", 9)[-1] == "空间里面他绝对赢不了的"
    assert ass_data[-1].split(",", 9)[-1] == "你们接着善后"


def test_out_only_txt(setup_and_teardown):
    save_dir, srt_path, ass_path, txt_path = setup_and_teardown

    img_dir = test_dir / "RGBImages"
    input_param = make_input(is_batch_rec=True, out_format="txt")
    extractor = RapidVideOCR(input_param)
    extractor(img_dir, save_dir)

    txt_data = read_txt(txt_path)
    assert len(txt_data) == 8
    assert txt_data[-2] == "你们接着善后"
