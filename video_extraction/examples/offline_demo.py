"""Synthetic end-to-end demo. No real contacts, credentials, or model requests."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from rapid_videocr.wechat_detail_extractor import (
    select_stable_detail_frames, analyze_selected_detail_frames,
    build_wechat_outputs, write_wechat_outputs,
)


class FakeClient:
    def __init__(self):
        self.count = 0

    def analyze_images(self, images, prompt, **kwargs):
        frames = []
        for index, _ in enumerate(images):
            department = ['示例公司/研发部', '示例公司/测试部'][self.count % 2]
            self.count += 1
            frames.append(dict(image_index=index, page_type='person', name='测试人员',
                               user_code='DEMO001', position='工程师', departments=[department]))
        return SimpleNamespace(content=json.dumps({'frames': frames}, ensure_ascii=False), raw={})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    video = root/'synthetic.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 4, (984,794))
    if not writer.isOpened():
        raise RuntimeError('MJPG video writer unavailable')
    try:
        for index in range(24):
            frame = np.full((794,984,3), 245, np.uint8)
            color = 210 if index < 12 else 150
            cv2.rectangle(frame, (440,90), (830,640), (color,color,color), -1)
            cv2.putText(frame, 'SYNTHETIC ONLY', (450,150), cv2.FONT_HERSHEY_SIMPLEX,
                        .8, (0,0,0), 2)
            writer.write(frame)
    finally:
        writer.release()
    selected, _ = select_stable_detail_frames(video,root)
    assert len(selected) == 2, len(selected)
    analyzed, failures = analyze_selected_detail_frames(root,selected,FakeClient(),workers=1,batch_size=2)
    assert not failures, failures
    outputs = build_wechat_outputs(video.name,analyzed)
    assert len(outputs['people']) == 1
    assert len(outputs['memberships']) == 2
    write_wechat_outputs(root,outputs)
    print('Offline demo passed: 2 stable frames, 1 synthetic person, 2 memberships; no model calls.')


if __name__ == '__main__':
    main()
