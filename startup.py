"""恢复已校准的企业微信窗口并导航；不启动录像、不自动登录。"""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from app import BASE, grab, read_image, save_image, select_rect
from vision import rows, arrow
from win_input import Windows


def match_entry(image, template):
    """匹配唯一图标/文字样本，兼容选中反色，不猜测未知布局。"""
    if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
        raise ValueError('入口样本大于搜索区域，请重新设置自动打开')
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    if gray.std() < 10:
        raise ValueError('入口样本没有足够图像特征')
    scores = np.abs(cv2.matchTemplate(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), gray,
                                     cv2.TM_CCOEFF_NORMED))
    _, score, _, (x, y) = cv2.minMaxLoc(scores)
    if score < .90:
        raise ValueError('入口或页面标志不匹配，请检查登录状态、缩放及布局')
    h, w = gray.shape
    other = scores.copy()
    other[max(0,y-h):y+h+1,max(0,x-w):x+w+1] = -1
    if other.max(initial=-1) >= .90:
        raise ValueError('入口图像不唯一，停止以免点击错误项目')
    return x+w//2, y+h//2


def validate_profile(profile, config):
    for key in ('window_rect', 'tree', 'recording', 'spacing'):
        if profile['layout'][key] != config[key]:
            raise ValueError('录屏校准已改变，请重新设置自动打开')
    left, top, right, bottom = config['window_rect']
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise ValueError('窗口布局无效，要求主屏内的固定窗口')
    for key in ('tree', 'recording'):
        x,y,w,h = config[key]
        if w <= 0 or h <= 0 or not (left <= x < x+w <= right and top <= y < y+h <= bottom):
            raise ValueError('校准区域超出窗口')
    for entry in profile['steps'] + [profile['ready']]:
        x,y,w,h = entry['region']
        if min(x,y) < 0 or min(w,h) <= 0 or x+w > right-left or y+h > bottom-top:
            raise ValueError('导航搜索区域超出窗口')


class Desktop(Windows):
    def __init__(self):
        super().__init__()
        self.u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        self.u.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        self.u.GetWindowTextW.argtypes = [wintypes.HWND,wintypes.LPWSTR,ctypes.c_int]
        self.u.SetForegroundWindow.argtypes = [wintypes.HWND]
        self.u.SetWindowPos.argtypes = [wintypes.HWND,wintypes.HWND,ctypes.c_int,ctypes.c_int,
                                        ctypes.c_int,ctypes.c_int,wintypes.UINT]
        self.k = ctypes.WinDLL('kernel32', use_last_error=True)
        self.k.OpenProcess.argtypes = [wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        self.k.OpenProcess.restype = wintypes.HANDLE
        self.k.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE,wintypes.DWORD,
                                                      wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD)]
        self.k.CloseHandle.argtypes = [wintypes.HANDLE]

    def executable(self, handle):
        pid = wintypes.DWORD()
        self.u.GetWindowThreadProcessId(handle, ctypes.byref(pid))
        process = self.k.OpenProcess(0x1000, False, pid.value)
        if not process:
            raise RuntimeError('无法核对企业微信进程')
        try:
            buffer = ctypes.create_unicode_buffer(32768)
            length = wintypes.DWORD(len(buffer))
            if not self.k.QueryFullProcessImageNameW(process, 0, buffer, ctypes.byref(length)):
                raise RuntimeError('无法读取企业微信程序路径')
            return buffer.value
        finally:
            self.k.CloseHandle(process)

    def find(self, executable):
        found = []
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL,wintypes.HWND,wintypes.LPARAM)
        def visit(handle, _):
            name = ctypes.create_unicode_buffer(256)
            title = ctypes.create_unicode_buffer(256)
            self.u.GetClassNameW(handle, name, 256)
            self.u.GetWindowTextW(handle, title, 256)
            if title.value == '企业微信' and any(s in name.value.lower() for s in ('wework','wxwork')):
                try:
                    if Path(self.executable(handle)) == Path(executable):
                        found.append(int(handle))
                except RuntimeError:
                    pass
            return True
        callback = callback_type(visit)
        self.u.EnumWindows.argtypes = [callback_type,wintypes.LPARAM]
        self.u.EnumWindows(callback, 0)
        if len(found) > 1:
            raise RuntimeError('发现多个企业微信主窗口，无法唯一选择')
        return found[0] if found else None

    def restore(self, handle, rect):
        l,t,r,b = rect
        if r > self.u.GetSystemMetrics(0) or b > self.u.GetSystemMetrics(1):
            raise RuntimeError('已保存窗口超出当前主屏，请重新校准')
        self.u.ShowWindow(handle, 9)
        if not self.u.SetWindowPos(handle, None,l,t,r-l,b-t,0x0040):
            raise RuntimeError('无法恢复窗口位置或尺寸')
        self.u.SetForegroundWindow(handle)


def open_window(desktop, profile, config, launch=None):
    """无窗口才启动；不会关闭已有企业微信或处理登录。"""
    handle = desktop.find(profile['executable'])
    if handle is None:
        if launch is None:
            launch = lambda path: subprocess.Popen([path], cwd=str(Path(path).parent))
        launch(profile['executable'])
        for _ in range(60):
            if (BASE/'STOP').exists() or desktop.u.GetAsyncKeyState(0x1B) & 0x8000:
                raise KeyboardInterrupt('用户停止')
            time.sleep(.5)
            handle = desktop.find(profile['executable'])
            if handle is not None:
                break
        if handle is None:
            raise RuntimeError('30 秒内未找到主窗口；请手动完成登录后重试')
    desktop.restore(handle, config['window_rect'])
    runtime = dict(config, hwnd=handle)
    time.sleep(1)
    desktop.checkpoint(runtime)
    return runtime


def navigate(desktop, profile, config, root=BASE, capture=grab):
    l,t,_,_ = config['window_rect']
    def locate(entry):
        desktop.checkpoint(config)
        x,y,w,h = entry['region']
        template = read_image(root/entry['template'])
        px,py = match_entry(capture([l+x,t+y,w,h]),template)
        return l+x+px,t+y+py
    for entry in profile['steps']:
        print(f"正在定位：{entry['template']}",flush=True)
        deadline = time.monotonic()+15
        while True:
            try:
                x,y = locate(entry)
                break
            except ValueError:
                if time.monotonic() >= deadline:
                    raise
                desktop.wait(.5,config)
        desktop.click(x,y,config)
        desktop.wait(2,config)
    deadline = time.monotonic()+15
    print('正在验证通讯录标题和组织树',flush=True)
    while True:
        try:
            locate(profile['ready'])
            image = capture(config['tree'])
            candidates = rows(image,config['spacing'])
            opened, closed = read_image(root/'opened.png'),read_image(root/'closed.png')
            if len(candidates) < 2 or not any(arrow(image,r,opened) or arrow(image,r,closed) for r in candidates):
                raise ValueError('未识别到可浏览组织树，不能确认导航成功')
            return len(candidates)
        except ValueError:
            if time.monotonic() >= deadline:
                raise
            desktop.wait(.5,config)


def prepare():
    config = json.loads((BASE/'config.json').read_text(encoding='utf-8'))
    profile = json.loads((BASE/'startup-profile.json').read_text(encoding='utf-8'))
    validate_profile(profile,config)
    if (BASE/'STOP').exists():
        raise KeyboardInterrupt('发现 STOP 文件')
    if not Path(profile['executable']).is_file():
        raise ValueError('企业微信路径不存在，请重新设置自动打开')
    desktop = Desktop()
    runtime = open_window(desktop,profile,config)
    count = navigate(desktop,profile,runtime)
    desktop.checkpoint(runtime)
    # 只有页面和树验证成功才更新句柄；保留原校准备份。
    original = (BASE/'config.json').read_bytes()
    (BASE/'config.before-prepare.json').write_bytes(original)
    temp = BASE/'config.prepare.tmp'
    temp.write_text(json.dumps(runtime,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(BASE/'config.json')
    print(f'已打开通讯录并恢复窗口，识别到 {count} 个可见节点。未开始录屏。')
    print('保留客户端当前滚动位置；新录制请先回到树顶，续录请保持断点画面。')


def setup():
    from PIL import ImageGrab
    config = json.loads((BASE/'config.json').read_text(encoding='utf-8'))
    desktop = Desktop()
    print('请保持已校准窗口，打开通讯录的全体组织树，5 秒后保存导航样本。',flush=True)
    time.sleep(5)
    desktop.checkpoint(config)
    screen = ImageGrab.grab()
    l,t,r,b = config['window_rect']
    entries = []
    for name,label in [('contacts','通讯录入口：紧框小图标，不含文字和鼠标'),
                       ('organization','全体/组织入口：紧框图标和文字'),
                       ('ready','组织树上方固定标题：不要框人员或人数')]:
        x,y,w,h = select_rect(screen,label)
        if not (l <= x < x+w <= r and t <= y < y+h <= b):
            raise ValueError('样本必须在企业微信窗口内')
        template = cv2.cvtColor(np.array(screen.crop((x,y,x+w,y+h))),cv2.COLOR_RGB2BGR)
        match_entry(template,template)
        filename = f'startup-{name}.png'
        save_image(BASE/filename,template)
        # 同一列上下搜索，允许侧栏条目纵向变化；页面标题局部核验。
        region = [max(0,x-l-8),0,min(w+16,r-x+8),b-t] if name != 'ready' else [x-l,y-t,w,h]
        entries.append(dict(template=filename,region=region))
    profile = dict(executable=desktop.executable(config['hwnd']),
                   layout={k:config[k] for k in ('window_rect','tree','recording','spacing')},
                   steps=entries[:2],ready=entries[2])
    validate_profile(profile,config)
    (BASE/'startup-profile.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2),encoding='utf-8')
    print('自动打开样本已保存。运行 startup.py prepare 验证。')


def main():
    parser = argparse.ArgumentParser(description='自动打开企业微信、恢复固定布局并进入通讯录；不录屏')
    parser.add_argument('command',choices=['setup','prepare'])
    args = parser.parse_args()
    try:
        setup() if args.command == 'setup' else prepare()
        return 0
    except (Exception,KeyboardInterrupt) as error:
        print(f'自动打开未完成：{error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
