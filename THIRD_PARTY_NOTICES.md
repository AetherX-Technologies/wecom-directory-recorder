# Third-party notices / 第三方说明

`video_extraction/rapid_videocr/` contains code derived from [SWHL/RapidVideOCR](https://github.com/SWHL/RapidVideOCR), licensed under Apache License 2.0. Original author/contact comments are preserved. The license is included in `video_extraction/LICENSE` and at the repository root. OpenCV, NumPy, Pillow, FFmpeg and other dependencies remain subject to their own licenses; their binaries and model weights are not redistributed here.

The imported local adaptation includes PaddleOCR-VL HTTP OCR, vision-service clients, screen-change selection, personnel extraction, identity normalization, conflict review, and a separate WeCom desktop-detail adapter. It is a local derivative, not a claim of endorsement or an unmodified upstream release.

Public integration changes:

- Package metadata uses a fixed version in `pyproject.toml`; upstream dynamic release/publishing scripts and workflows are not imported.
- Test identities are fictionalized. Legacy subtitle images are generated during tests instead of distributing original screenshots or media.
- `personnel_finalizer.py` has empty default name/department correction and evidence mappings. Dataset-specific decisions are not transferable algorithm defaults. Its functions remain available for reviewed local rules.
- `screen_ocr_all_frames.py` no longer embeds a personal watermark or date. Optional `OCR_WATERMARK_PATTERN` and `OCR_WATERMARK_DATE_PATTERN` environment variables enable user-supplied regex filters; empty settings match nothing.
- Changed imported Python files contain an integration notice. The main WeCom detail extraction algorithm and model/provider restrictions are retained.
- Original local project files, Git history, recordings, extracted data, credentials, and deployment configuration are not copied into this public repository.

公开版本保留第三方署名与许可证。隐私处理只发生在本仓库副本，原视频识别项目保持不变；仓库可公开并不表示输入视频或识别结果可以公开。
