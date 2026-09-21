# 技术选型与检索记录

检索日期：2026-09-20。采用 Exa 搜索与 literature-review 技能做针对性技术范围检索，不宣称系统综述。

问题：Windows 企业微信组织树如何以最低推理开销逐项浏览并为离线 OCR 录屏？比较控件自动化、图像匹配 RPA、通用智能 GUI Agent；关注依赖体积、CPU、版本适配、滚动不漏项。纳入开源项目主仓库/官方文档和 GUI 截图自动化原始论文；不将普通微信项目视为企业微信兼容证明。未下载或执行搜索结果代码。

## 对比

| 方案 | 已查证依据 | 对本需求的判断 |
|---|---|---|
| [pywinauto](https://github.com/pywinauto/pywinauto) | 主仓库说明支持 Win32/UI Automation 与鼠标键盘 | 控件树完整时很适合，低计算开销；当前企业微信控件可访问性未验证 |
| [RPA Framework Windows](https://github.com/robocorp/rpaframework/blob/70233655/packages/windows/src/RPA/Windows/__init__.py) | 文档说明基于 uiautomation，Windows 库本身不支持图像定位 | 能搭流程，但完整框架不是本需求的必要依赖 |
| [SikuliX](https://github.com/RaiMan/SikuliX1) | 项目内容描述基于 OpenCV 截图匹配，无需应用控件接口 | 方法最贴近本需求；使用现成 IDE 仍需自行编写树遍历和滚动校验 |
| 本目录轻量程序 | Python、OpenCV、Pillow、FFmpeg | 采用同类图像匹配思路，没有 Java/RPA 平台/模型服务；只适配明确的界面样式 |

本轮未找到有可验证证据、能直接覆盖截图所示企业微信版本的完整通讯录录屏开源成品。搜索中的普通微信自动化项目和泛化博客不作为兼容性证据。

## 论文证据

Tom Yeh, Tsung-Hsiang Chang, Robert C. Miller. **Sikuli: Using GUI Screenshots for Search and Automation**, ACM UIST 2009. DOI：[10.1145/1622176.1622213](https://doi.org/10.1145/1622176.1622213)。[MIT 原文](https://dspace.mit.edu/server/api/core/bitstreams/64eeae27-71f5-400c-a17b-91bf325e9d3e/content)。会议论文，非预印本。

通过 DOI 搜索结果核对作者、年份和标识；Exa 返回的 MIT 原文内容说明，小图标采用模板匹配可以定位并驱动鼠标键盘，大图案可使用局部特征。该文为采用截图匹配提供技术依据，**没有测试本企业微信版本，也不能证明通讯录遍历完整率或本机 CPU 占用**。论文中的搜索用户研究不能误当作本程序遍历成功率。

## 搜索日志与限制

来源渠道：Exa 的网页索引，包括 GitHub、官方项目说明、MIT 论文正文及 DOI 页面；没有直接调用 Semantic Scholar/Crossref/arXiv 数据库 API。

| 查询 | 返回条目数 | 使用方式 |
|---|---:|---|
| `pywinauto WeCom 企业微信 通讯录 UI Automation 树` | 5 | 排除泛化博客、普通微信兼容性推断 |
| `Sikuli using GUI screenshots visual interface automation UIST 2009 paper` | 3 | 定位 DOI 与原始论文 |
| `github RPA Windows desktop automation UIAutomation RPA Framework pywinauto` | 4 | 主仓库与 Windows 库文档 |
| `site.arxiv.org GUI automation template matching Sikuli desktop automation` | 3 | 返回 MIT/SikuliX，未获直接 arXiv 证据 |

搜索结果中同一 Sikuli 论文和仓库重复出现，按 DOI/规范 URL 合并。仅保留一篇直接相关原始论文，不声称跨论文效果共识。官方 pywinauto README 与 SikuliX README 另行 Exa 抓取核对；CSAIL 旧 PDF 地址抓取返回 HTTP 500，改用已返回的 MIT 机构库原文。SikuliX 抓取内容包含迁移描述，未进一步审计继任项目，不据此承诺其版本、维护状态或兼容性。
