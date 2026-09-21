"""录屏助手自身的 Windows 输入与停止保护。"""
import ctypes
from ctypes import wintypes
import time
from pathlib import Path


class Windows:
    def __init__(self):
        self.u = ctypes.WinDLL('user32', use_last_error=True)
        self.u.SetProcessDPIAware()
        self.u.GetForegroundWindow.restype = wintypes.HWND
        self.u.WindowFromPoint.argtypes = [wintypes.POINT]
        self.u.WindowFromPoint.restype = wintypes.HWND
        self.u.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        self.u.GetAncestor.restype = wintypes.HWND
        self.u.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.u.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        self.paused = False
        self.was_f8 = False

    def geometry(self, handle):
        rect = wintypes.RECT()
        if not self.u.GetWindowRect(handle, ctypes.byref(rect)):
            raise RuntimeError('目标窗口已关闭')
        return [rect.left, rect.top, rect.right, rect.bottom]

    def target_at(self, x, y):
        handle = self.u.GetAncestor(self.u.WindowFromPoint(wintypes.POINT(x, y)), 2)
        name = ctypes.create_unicode_buffer(256)
        self.u.GetClassNameW(handle, name, 256)
        if not any(token in name.value.lower() for token in ('wework', 'wxwork')):
            raise RuntimeError(f'所选区域不在企业微信主窗口中（窗口类：{name.value}）')
        return int(handle), self.geometry(handle)

    def checkpoint(self, config):
        stalled = 0.0
        while True:
            if (Path(__file__).resolve().parent/'STOP').exists():
                raise KeyboardInterrupt('发现 STOP 文件，停止')
            if self.u.GetAsyncKeyState(0x1B) & 0x8000:
                raise KeyboardInterrupt('按 Esc 停止')
            f8 = bool(self.u.GetAsyncKeyState(0x77) & 0x8000)
            if f8 and not self.was_f8:
                self.paused = not self.paused
                print('已暂停；F8 继续，Esc 停止' if self.paused else '继续', flush=True)
            self.was_f8 = f8
            if not self.paused:
                break
            time.sleep(.05)
        foreground = int(self.u.GetForegroundWindow() or 0)
        started = time.monotonic()
        announced = False
        while foreground != config['hwnd']:
            name = ctypes.create_unicode_buffer(256)
            self.u.GetClassNameW(foreground, name, 256)
            if name.value != 'Ghost' or time.monotonic()-started >= 30:
                break
            if not announced:
                print('企业微信暂时无响应，暂停输入等待恢复（最多 30 秒）', flush=True)
                announced = True
            if (Path(__file__).resolve().parent/'STOP').exists() or self.u.GetAsyncKeyState(0x1B) & 0x8000:
                raise KeyboardInterrupt('等待恢复时收到停止请求')
            point = wintypes.POINT()
            self.u.GetCursorPos(ctypes.byref(point))
            if point.x <= 2 and point.y <= 2:
                raise KeyboardInterrupt('鼠标移到屏幕左上角，停止')
            f8 = bool(self.u.GetAsyncKeyState(0x77) & 0x8000)
            if f8 and not self.was_f8:
                self.paused = not self.paused
            self.was_f8 = f8
            time.sleep(.1)
            foreground = int(self.u.GetForegroundWindow() or 0)
        if announced:
            stalled = time.monotonic()-started
        if foreground != config['hwnd']:
            name = ctypes.create_unicode_buffer(256)
            self.u.GetClassNameW(foreground, name, 256)
            raise RuntimeError(f'企业微信失去前台焦点，已停止（前台句柄 {foreground}，窗口类 {name.value}）')
        if self.paused:
            self.checkpoint(config)
            stalled = time.monotonic()-started
        if self.geometry(config['hwnd']) != config['window_rect']:
            raise RuntimeError('窗口位置或尺寸改变，必须重新校准')
        point = wintypes.POINT()
        self.u.GetCursorPos(ctypes.byref(point))
        if point.x <= 2 and point.y <= 2:
            raise KeyboardInterrupt('鼠标移到屏幕左上角，停止')
        return stalled

    def wait(self, seconds, config, recorder=None):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            end += self.checkpoint(config)
            if recorder:
                recorder.check()
            time.sleep(.05)

    def click(self, x, y, config):
        self.checkpoint(config)
        target = self.u.GetAncestor(self.u.WindowFromPoint(wintypes.POINT(x, y)), 2)
        if int(target or 0) != config['hwnd']:
            raise RuntimeError('点击位置被其他窗口遮挡，停止')
        self.u.SetCursorPos(x, y)
        self.u.mouse_event(0x0002, 0, 0, 0, 0)
        self.u.mouse_event(0x0004, 0, 0, 0, 0)

    def scroll(self, x, y, config):
        self.checkpoint(config)
        self.u.SetCursorPos(x, y)
        notches = int(config.get('scroll_notches',1))
        if not 1 <= notches <= 6:
            raise ValueError('scroll_notches 必须为 1–6')
        # 当前企业微信把单个高 delta 事件仍当一次小滚动；逐个发送刻度。
        for _ in range(notches):
            self.checkpoint(config)
            self.u.mouse_event(0x0800, 0, 0, ctypes.c_ulong(-120).value, 0)
            time.sleep(.06)
