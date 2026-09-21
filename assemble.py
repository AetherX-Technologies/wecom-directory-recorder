"""按已校验的续录链合并视频与逐行索引；不从名字去重。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import hashlib
from audit_run import audit


def reviewed_tail(folder, report):
    """仅允许有哈希绑定复核记录的失败点击尾段缺帧；原始审计保持失败。"""
    path=folder/'tail-review.json'
    if not path.exists():
        return None
    review=json.loads(path.read_text(encoding='utf-8'))
    failed={name for name,ok in report['checks'].items() if not ok}
    if failed!={'video_duration_matches_run'} or report['run_status']!='blocked':
        return None
    if review.get('decision')!='confirmed_visits_present_unconfirmed_tail_missing':
        return None
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    duration=float(report['media']['format']['duration'])
    events=[json.loads(s) for s in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    last=[e for e in events if e['kind']=='visited'][-1]
    if not (last['elapsed']<=duration<summary['elapsed_seconds']<=duration+10):
        return None
    required={'capture.mkv','events.jsonl','summary.json','tail-frame.png',f"{last['index']:06d}.png"}
    hashes=review.get('sha256',{})
    if set(hashes)!=required:
        return None
    for name,digest in hashes.items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:
            return None
    return review


def assemble(last):
    chain=[]
    seen=set()
    current=Path(last).resolve()
    exceptions=[]
    while True:
        if current in seen:
            raise ValueError('续录链包含循环')
        seen.add(current)
        config=json.loads((current/'config.json').read_text(encoding='utf-8'))
        if audit(current):
            report=json.loads((current/'audit.json').read_text(encoding='utf-8'))
            review=reviewed_tail(current,report)
            if not review:
                raise ValueError(f'产物校验失败且没有适用的尾段复核：{current}')
            exceptions.append(dict(segment=str(current),review=review))
        chain.append(current)
        parent=config.get('resume_from')
        if not parent:
            break
        current=Path(parent).resolve()
    chain.reverse()
    output=Path(__file__).resolve().parent/'delivery'/datetime.now().strftime('%Y%m%d-%H%M%S')
    output.mkdir(parents=True)
    lines=[]
    elapsed=0.0
    index=[]
    shapes=set()
    for segment in chain:
        report=json.loads((segment/'audit.json').read_text(encoding='utf-8'))
        stream=report['media']['streams'][0]
        shapes.add((stream['codec_name'],stream['width'],stream['height'],stream['avg_frame_rate']))
        path=(segment/'capture.mkv').as_posix().replace("'", "'\\''")
        lines.append("file '"+path+"'")
        for event in map(json.loads,(segment/'events.jsonl').read_text(encoding='utf-8').splitlines()):
            if event['kind']=='visited':
                index.append(dict(index=len(index)+1,segment=segment.name,local_index=event['index'],
                                  video_seconds=round(elapsed+event['elapsed'],3),
                                  kind_hint=event['kind_hint'],
                                  png=str(segment/f"{event['index']:06d}.png")))
        elapsed+=float(report['media']['format']['duration'])
    if len(shapes)!=1:
        raise ValueError('各段编码/画幅不同，拒绝无损拼接')
    concat=output/'concat.txt'
    concat.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','concat','-safe','0',
                    '-i',str(concat),'-c','copy',str(output/'capture-all.mkv')],check=True)
    (output/'frames.jsonl').write_text('\n'.join(json.dumps(e,ensure_ascii=False) for e in index)+'\n',encoding='utf-8')
    last_summary=json.loads((chain[-1]/'summary.json').read_text(encoding='utf-8'))
    (output/'manifest.json').write_text(json.dumps(dict(segments=[str(p) for p in chain],
        visited_rows=len(index),estimated_duration_seconds=elapsed,last_status=last_summary['status'],
        reviewed_exceptions=exceptions,
        coverage_verified=False,notes='合并产物和顺序索引；不等于唯一人数或接口级完整性验证。'),
        ensure_ascii=False,indent=2),encoding='utf-8')
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='合并从头录制并逐段校验续录的录像链。')
    parser.add_argument('last_segment')
    assemble(parser.parse_args().last_segment)
