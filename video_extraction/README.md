# 视频信息提取模块

[中文端到端用法](../README.md) · [English guide](../README.en.md) · [来源与修改说明](../THIRD_PARTY_NOTICES.md)

从仓库根目录安装：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .\video_extraction
```

推荐桌面详情入口：`python -m rapid_videocr.wechat_detail_extractor`。使用 `--selection-only` 先离线检查稳定帧，再配置根目录 `.env` 的 `VL_MODEL_*` 执行模型分析。当前 ROI 为指定树与详情画面的比例裁剪，不自动适配任意录像。

## 保留的其他算法

| 模块 | 用途 |
|---|---|
| `wechat_detail_extractor` | 桌面个人详情 → 工号人员表、部门关系表、职务表与复核项 |
| `personnel_extractor` | 旧版竖屏名单布局 → 姓名、职务、部门路径观察记录；不是桌面录屏适配器 |
| `personnel_identity_finalizer` | 已匹配工号的 CSV → 一人一行、人员部门关系及复核输出 |
| `personnel_finalizer` / `conflict_verifier` | 旧观察表清理与冲突复核；公开版不带私有人工修正规则 |
| `screen_ocr` / `screen_ocr_all_frames` | PaddleOCR-VL HTTP 文本提取与帧筛选 |
| `main` / `vsf_ocr_cli` / `vsf_cli` | 原 RapidVideOCR 字幕及 VideoSubFinder 工具 |

各模块支持 `python -m rapid_videocr.<模块名> --help`。旧版名单布局与当前桌面详情布局不能混用。

身份归并模块独立用法：

```powershell
python -m rapid_videocr.personnel_identity_finalizer `
  --matches .\local-only\matches.csv `
  --overrides .\local-only\memberships-reviewed.csv `
  --review .\local-only\identity-review.csv `
  --output-root .\outputs\identity
```

匹配表要求 `user_code,user_name,title,sso_dept_path`。人工关系覆盖表使用这些列及 `override_mode`，`replace_user` 会替换该工号的全部关系，只有掌握完整证据时才用；这与桌面详情命令的姓名/职务 `--overrides` 格式不同。此模块不从视频自动获得工号匹配数据。

模型接口返回、画面变化阈值、人工覆盖及去重均有明确局限，详见首页。不要把模型识别出的合法格式工号直接当作经过后台验证的身份。
