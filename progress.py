"""只读汇总当前连续录制链，不把行数换算成企业人数。"""
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent


def read_json(path,default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError,json.JSONDecodeError):
        return default


def main():
    batch=read_json(BASE/'batch-status.json',{})
    current=batch.get('current')
    if not current:
        print(json.dumps(batch,ensure_ascii=False,indent=2))
        return
    seen=set(); chain=[]
    while current:
        folder=Path(current).resolve()
        if folder in seen:
            raise ValueError('续录链循环')
        seen.add(folder)
        config=read_json(folder/'config.json',{})
        events=[]
        try:
            for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # 正在写入的末条事件留到下次
        except FileNotFoundError:
            pass
        visits=[e for e in events if e['kind']=='visited']
        summary=read_json(folder/'summary.json')
        chain.append(dict(folder=str(folder),visited_rows=len(visits),
                          person_candidate_rows=sum(e['kind_hint']=='person_or_unmatched' for e in visits),
                          department_rows=sum(e['kind_hint']=='department' for e in visits),
                          last_event=events[-1] if events else None,
                          status=summary['status'] if summary else 'running_or_not_finalized'))
        current=config.get('resume_from')
    report=dict(batch_state=batch.get('state'),total_visited_rows=sum(s['visited_rows'] for s in chain),
                total_person_candidate_rows=sum(s['person_candidate_rows'] for s in chain),
                total_department_rows=sum(s['department_rows'] for s in chain),segments=list(reversed(chain)))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
