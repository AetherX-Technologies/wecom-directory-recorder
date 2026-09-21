"""离线核对录屏时长、事件连续性和 PNG，不操作桌面。"""
import argparse
import json
from pathlib import Path
import subprocess
import hashlib
import cv2
import numpy as np
from vision import Row,selected,arrow


def audit(folder):
    folder=Path(folder)
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    visits=[e for e in events if e['kind']=='visited']
    scrolls=[e for e in events if e['kind']=='scroll']
    probe=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration,size',
        '-show_entries','stream=codec_name,width,height,avg_frame_rate','-of','json',str(folder/'capture.mkv')],
        capture_output=True,text=True,check=True)
    media=json.loads(probe.stdout)
    duration=float(media['format']['duration'])
    count=summary['visited_rows']
    pngs=sorted(folder.glob('[0-9][0-9][0-9][0-9][0-9][0-9].png'))
    config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    invalid_images=[]
    unchanged_person_detail_pairs=[]
    previous_detail=None
    previous_visit=None
    template_names=('closed.png','opened.png','folder.png')
    template_base=folder if all((folder/name).exists() for name in template_names) else Path(__file__).resolve().parent
    templates={name:cv2.imdecode(np.fromfile(template_base/name,dtype=np.uint8),cv2.IMREAD_COLOR) for name in template_names}
    if any(value is None for value in templates.values()):
        raise ValueError('审计所需箭头/文件夹模板无法读取')
    department_state_issues=[]
    if pngs:
        rx,ry,rw,rh=config['recording']
        tx,ty,tw,th=config['tree']
        for event in visits:
            image_path=folder/f"{event['index']:06d}.png"
            try:
                image=cv2.imdecode(np.fromfile(image_path,dtype=np.uint8),cv2.IMREAD_COLOR)
                if image is None or image.shape[:2]!=(rh,rw):
                    raise ValueError('PNG 解码或尺寸不符')
                tree=image[ty-ry:ty-ry+th,tx-rx:tx-rx+tw]
                y=event['y']
                if not selected(tree,Row(y,max(0,y-18),min(th,y+18))):
                    raise ValueError('PNG 中目标行不是蓝色选中状态')
                row=Row(y,max(0,y-18),min(th,y+18))
                if event['kind_hint']=='department':
                    if arrow(tree,row,templates['closed.png']) or not arrow(tree,row,templates['opened.png']):
                        department_state_issues.append(dict(index=event['index'],reason='部门未确认保持展开'))
                elif arrow(tree,row,templates['folder.png'],.88):
                    department_state_issues.append(dict(index=event['index'],reason='人员候选中检测到文件夹'))
                detail=image[:,tx-rx+tw+20:]
                digest=hashlib.blake2b(detail.tobytes(),digest_size=16).hexdigest()
                if (previous_visit and previous_visit['kind_hint']=='person_or_unmatched'
                    and event['kind_hint']=='person_or_unmatched' and digest==previous_detail):
                    unchanged_person_detail_pairs.append([previous_visit['index'],event['index']])
                previous_detail,previous_visit=digest,event
            except (OSError,ValueError,cv2.error) as error:
                invalid_images.append(dict(index=event['index'],reason=str(error)))
    checks={
        'visit_indices_contiguous':[e['index'] for e in visits]==list(range(1,count+1)),
        'event_times_monotonic':all(a['elapsed']<=b['elapsed'] for a,b in zip(events,events[1:])),
        'video_contains_last_visit':bool(visits) and duration+1>=visits[-1]['elapsed'],
        'video_duration_matches_run':abs(duration-summary['elapsed_seconds'])<3,
        'snapshots_contiguous':(not pngs and not config.get('snapshots')) or [p.stem for p in pngs]==[f'{i:06d}' for i in range(1,count+1)],
        'recording_finished':summary['status']!='recording_error',
        'png_selected_rows_verified':not invalid_images,
        'department_states_verified':not department_state_issues,
    }
    report=dict(checks=checks,artifact_checks_pass=all(checks.values()),
                visited_rows=count,person_candidate_rows=sum(e['kind_hint']=='person_or_unmatched' for e in visits),
                department_rows=sum(e['kind_hint']=='department' for e in visits),
                scroll_events=len(scrolls),snapshot_count=len(pngs),media=media,
                invalid_images=invalid_images,unchanged_person_detail_pairs=unchanged_person_detail_pairs,
                department_state_issues=department_state_issues,
                template_source=str(template_base.resolve()),
                templates_saved_with_segment=template_base==folder,
                template_sha256={name:hashlib.sha256((template_base/name).read_bytes()).hexdigest() for name in template_names},
                detail_review_needed=bool(unchanged_person_detail_pairs),
                run_status=summary['status'],coverage_verified=False)
    (folder/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['artifact_checks_pass'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='检查已结束的录屏产物；不证明通讯录人数或全量覆盖。')
    parser.add_argument('folder')
    raise SystemExit(audit(parser.parse_args().folder))
