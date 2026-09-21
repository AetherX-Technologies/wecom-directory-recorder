"""有界分段录制：只在上一段正常达到上限且产物通过审计时自动续录。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from audit_run import audit

BASE=Path(__file__).resolve().parent


def read_summary(folder):
    try:
        return json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    except (FileNotFoundError,json.JSONDecodeError):
        return None


def main(args):
    previous=Path(args.after).resolve() if args.after else None
    status_path=BASE/'batch-status.json'
    started=time.monotonic()
    segments=[]
    def status(state,**fields):
        value=dict(state=state,updated=datetime.now().isoformat(),segments=segments,**fields)
        temp=status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
        temp.replace(status_path)
    if previous:
        status('waiting_existing_segment',current=str(previous))
        while read_summary(previous) is None:
            if time.monotonic()-started>args.max_hours*3600:
                raise TimeoutError('等待现有录制段结束超时')
            if (BASE/'STOP').exists():
                status('stopped',reason='检测到 STOP 文件')
                return 1
            time.sleep(2)
    for _ in range(args.max_segments):
        if (BASE/'STOP').exists():
            status('stopped',reason='检测到 STOP 文件')
            return 1
        if previous:
            summary=read_summary(previous)
            if summary is None:
                raise RuntimeError('上一段没有正常写入结束状态')
            if audit(previous):
                status('blocked',current=str(previous),reason='产物审计失败')
                return 1
            if summary['status']=='end_candidate':
                status('end_candidate',current=str(previous),reason='已到界面末尾候选，等待核对后交付')
                return 0
            if summary['status']!='limit_reached':
                status('blocked',current=str(previous),reason=summary['reason'])
                return 1
        if time.monotonic()-started>args.max_hours*3600:
            status('limit_reached',reason='分段录制总时间达到上限')
            return 1
        current=BASE/'runs'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        command=[sys.executable,str(BASE/'app.py'),'run','--limit',str(args.segment_rows),
                 '--snapshots','--output-dir',str(current)]
        if previous:
            command.extend(['--resume',str(previous)])
        segments.append(str(current))
        status('recording',current=str(current),previous=str(previous) if previous else None)
        print(f'开始分段：{current}',flush=True)
        # 输入、录屏与停止逻辑都由录屏程序负责；主管不强行恢复焦点或错误画面。
        result=subprocess.run(command,cwd=BASE)
        if result.returncode or read_summary(current) is None:
            status('blocked',current=str(current),reason='本段失败；保留日志，等待定位')
            return 1
        previous=current
    status('limit_reached',current=str(previous),reason='分段数量达到上限')
    return 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='自动分段、审计、续录。异常或用户停止时不自行重试。')
    parser.add_argument('--after',help='接续已完成或正在运行的上一段目录')
    parser.add_argument('--segment-rows',type=int,default=500)
    parser.add_argument('--max-segments',type=int,default=60)
    parser.add_argument('--max-hours',type=float,default=12)
    options=parser.parse_args()
    if not 1<=options.segment_rows<=10000 or not 1<=options.max_segments<=60 or not 0<options.max_hours<=24:
        parser.error('每段行数 1–10000，分段数 1–60，总时长 0–24 小时')
    raise SystemExit(main(options))
