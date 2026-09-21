"""纯图像算法；不读取桌面，不控制鼠标。坐标均相对通讯录区域。"""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class Row:
    y: int
    top: int
    bottom: int


def selected(image, row):
    """当前浅色主题的蓝色整行选中状态；主题不匹配时拒绝继续。"""
    # 长部门名会填满中央文字带，需包含上下空白边缘；范围不跨到相邻行。
    region = image[max(0,row.y-13):min(len(image),row.y+14), :-10]
    b, g, r = cv2.split(region)
    blue = (b > 170) & (g > 90) & (r < 135) & ((b.astype(int)-r) > 60)
    return bool(np.count_nonzero(np.mean(blue,axis=1) > .65) >= 3)


def rows(image, spacing=36, footer_template=None):
    # 深色文字/头像、蓝色文件夹。浅灰水印不应进入候选。
    b, g, r = cv2.split(image[:, :-10])
    dark = (np.maximum(np.maximum(b, g), r) < 170)
    blue = (b > 170) & (g > 90) & (r < 135) & ((b.astype(int)-r) > 60)
    active = np.count_nonzero(dark | blue, axis=1) >= 4
    # 整行选中背景与上一行图标/鼠标轮廓只隔几像素，不能按普通文字间隙合并。
    # 用宽蓝色带确定独立选中行，内部白字造成的空隙仍属于同一行。
    wide = np.flatnonzero(np.mean(blue, axis=1) > .65)
    selection = None
    if len(wide) >= 3:
        a, z = int(wide[0]), int(wide[-1])+1
        if z-a <= spacing:
            selection = (a, z)
    indices = np.flatnonzero(active)
    if not len(indices):
        return []
    cuts = np.diff(indices) > max(4, spacing // 6)
    if selection:
        a, z = selection
        cuts |= ((indices[:-1] < a) & (indices[1:] >= a)) | ((indices[:-1] < z) & (indices[1:] >= z))
        cuts &= ~((indices[:-1] >= a) & (indices[1:] < z))
    groups = np.split(indices, np.where(cuts)[0]+1)
    result = []
    for group in groups:
        a, z = int(group[0]), int(group[-1])+1
        if z-a < 7:
            continue
        if z-a > spacing:
            raise ValueError('行图像粘连：请调整树区域或行距，停止以免漏项')
        y = (a+z)//2
        if a < 2 or z >= len(image)-1:
            continue  # 实际文字/图标在边缘被裁断的行留给下一屏
        # 最底行的空白边距可能被窗口裁掉，不能因此漏掉文字完整的最后一人。
        result.append(Row(y, max(0,y-spacing//2), min(len(image),y+spacing//2)))
    if footer_template is not None:
        # 只排除实机核对过的页脚模板，且必须位于树底部一行、水平居中。
        fh, fw = footer_template.shape[:2]
        region = image[-spacing:, :-10]
        if fh <= len(region) and fw <= region.shape[1]:
            scores = cv2.matchTemplate(region, footer_template, cv2.TM_CCOEFF_NORMED)
            _, score, _, point = cv2.minMaxLoc(scores)
            x, y = point
            center = len(image)-spacing+y+fh//2
            if score >= .98 and abs(x+fw/2-image.shape[1]/2) <= image.shape[1]*.12:
                result = [row for row in result if abs(row.y-center) > fh/2 or selected(image,row)]
    if any(b.y-a.y < spacing*0.65 for a, b in zip(result, result[1:])):
        raise ValueError('行间距异常，不能可靠区分节点')
    return result


def arrow(image, row, template, threshold=0.91):
    region = image[row.top:row.bottom, :image.shape[1]-16]
    if region.shape[0] < template.shape[0] or region.shape[1] < template.shape[1]:
        return None
    # 灰度匹配减少选中背景影响；低置信度不作为点击依据。
    scores = cv2.matchTemplate(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)
    # 选中后灰色箭头变成浅色，灰度对比方向反转；取绝对相关。
    _, score, _, point = cv2.minMaxLoc(np.abs(scores))
    if score < threshold:
        return None
    return point[0]+template.shape[1]//2, row.top+point[1]+template.shape[0]//2, score


def locate_anchor(image, anchor, old_y, threshold=0.96, center_offset=None, allow_avatar_change=False):
    width = anchor.shape[1]
    scores = cv2.matchTemplate(image[:, :width], anchor, cv2.TM_CCOEFF_NORMED).reshape(-1)
    best = int(np.argmax(scores))
    if scores[best] < threshold:
        if allow_avatar_change:
            return locate_selected_name(image,anchor,old_y,center_offset)
        raise ValueError('滚动衔接图像丢失；已停止，不能保证无漏项')
    other = scores.copy()
    other[max(0, best-anchor.shape[0]):best+anchor.shape[0]+1] = -1
    if other.max(initial=-1) > threshold:
        raise ValueError('滚动衔接存在多个相同图像；已停止，避免跳过同名节点')
    y = best+(anchor.shape[0]//2 if center_offset is None else center_offset)
    if y > old_y+3:
        raise ValueError('滚动方向异常')
    return y


def locate_selected_name(image, anchor, old_y, center_offset=None):
    """仅供已选中人员：跳过首个头像，用白色姓名笔画验证唯一对应。"""
    center=anchor.shape[0]//2 if center_offset is None else center_offset
    if center<13 or center+10>len(anchor) or not selected(anchor,Row(center,0,len(anchor))):
        raise ValueError('头像变化恢复需要完整的已选中人员行')
    foreground=np.max(np.abs(anchor[center-9:center+10].astype(int)-anchor[center-13:center-12].astype(int)),axis=2)>45
    cols=np.flatnonzero((foreground.sum(axis=0)>=3)&(np.arange(anchor.shape[1])>10))
    if not len(cols):
        raise ValueError('姓名锚点缺少头像与文字')
    start=int(cols[0])+24
    width=anchor.shape[1]
    if width-start<24:
        raise ValueError('姓名锚点可见宽度不足')
    template=(np.min(anchor[center-9:center+10,start:],axis=2)>180).astype(np.uint8)*255
    if np.count_nonzero(template)<30 or np.count_nonzero(template)>template.size*.5:
        raise ValueError('姓名锚点缺少有效笔画')
    target=(np.min(image[:,start:width],axis=2)>180).astype(np.uint8)*255
    scores=cv2.matchTemplate(target,template,cv2.TM_CCOEFF_NORMED).ravel()
    best=int(np.argmax(scores))
    if scores[best]<.94:
        raise ValueError('姓名锚点不匹配，停止以免跳过人员')
    other=scores.copy()
    other[max(0,best-len(anchor)):best+len(anchor)+1]=-1
    if other.max(initial=-1)>.90:
        raise ValueError('姓名锚点不唯一，停止以免跳过同名节点')
    y=best+9
    if y>old_y+3 or not selected(image,Row(y,max(0,y-18),min(len(image),y+18))):
        raise ValueError('姓名锚点方向或选中状态不符')
    return y
