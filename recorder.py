"""独立 FFmpeg 低帧率录屏，保留进程错误日志。"""
import shutil
import subprocess
import time


class Recorder:
    def __init__(self, directory, rect, executable='ffmpeg', fps=4):
        binary = shutil.which(executable)
        if not binary:
            raise RuntimeError('找不到 FFmpeg，请安装并加入 PATH，或在配置中填写 ffmpeg 完整路径')
        x, y, w, h = rect
        self.log = (directory/'ffmpeg.log').open('w', encoding='utf-8')
        args = [binary, '-hide_banner', '-loglevel', 'warning', '-nostats', '-n',
                '-f', 'gdigrab', '-framerate', str(fps), '-draw_mouse', '0',
                '-offset_x', str(x), '-offset_y', str(y), '-video_size', f'{w}x{h}',
                '-i', 'desktop', '-an', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18', '-threads', '2',
                '-pix_fmt', 'yuv420p', str(directory/'capture.mkv')]
        try:
            self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                         stderr=self.log, creationflags=0x08000000)
            time.sleep(1)
            self.check()
        except BaseException:
            self.log.close()
            raise

    def check(self):
        if self.proc.poll() is not None:
            raise RuntimeError('录屏进程已退出，查看 ffmpeg.log；自动点击已停止')

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.communicate(b'q\n', timeout=15)
            if self.proc.returncode != 0:
                raise RuntimeError('录屏未正常结束，请检查 ffmpeg.log 和视频完整性')
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
            raise RuntimeError('录屏结束超时；保留 MKV 供检查')
        finally:
            self.log.close()
