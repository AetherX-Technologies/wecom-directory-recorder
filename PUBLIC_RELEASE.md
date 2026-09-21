# 公开整合验证 / Public integration validation

2026-09-21，Windows / Python 3.13。

- 在独立 `.venv-public` 安装 `requirements-dev.txt`，并用 `pip install -e ./video_extraction` 验证本地包安装。
- 录屏测试：44 项通过。解析测试：61 项通过，测试期间禁止网络连接；未调用真实模型或使用私有录像。
- CLI 帮助可运行。生成 6 秒、4 FPS 的虚构视频，完整执行稳定帧选择、固定模型替身分析、人员归并和 CSV 导出：2 个稳定帧、1 名虚构人员、2 条部门关系。
- 模型替身最初遗漏响应的 `raw` 字段，离线样例暴露问题后已修正，并用新输出目录重跑成功。不是服务端真实识别测试。
- 新建公开中文首页 `README.md`、英文说明 `README.en.md`，保留录屏细节于 `RECORDER_GUIDE.md`；补充模型依赖、裁剪比例、限量试跑、结果复核及来源说明。
- 原视频识别项目保持原位置；仅复制算法与测试，不合并它的 Git 历史、录像、图片、人员表或 `.env`。公开副本采用虚构身份样例，移除私有姓名/部门修正默认值及个人水印规则，保留第三方署名和许可证。

These checks validate installation and offline pipeline behavior, not model accuracy, arbitrary WeCom layouts, complete directory coverage, or a fresh cold-start desktop session. The public repository deliberately contains no real directory dataset or calibration templates.

GitHub Actions includes the same offline test commands. Local passing results do not imply that a remote CI run has already passed.
