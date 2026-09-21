"""企业微信可视通讯录录屏助手。仅在用户本机启动时控制桌面。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image, ImageGrab, ImageTk
from vision import rows, arrow, locate_anchor, selected

BASE = Path(__file__).resolve().parent


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'无法读取图像：{path}')
    return image


def save_image(path, image):
    ok, data = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError('PNG 编码失败')
    data.tofile(path)


def grab(rect):
    x, y, w, h = rect
    return cv2.cvtColor(np.array(ImageGrab.grab(bbox=(x, y, x+w, y+h))), cv2.COLOR_RGB2BGR)


def settled_tree(capture, wait, samples=30):
    """滚动后额外观察至少五秒，连续两秒画面稳定才继续定位。"""
    previous=capture()
    stable=0
    for index in range(samples):
        wait(.5)
        current=capture()
        difference=float(np.mean(np.abs(current.astype(float)-previous.astype(float))))
        stable=stable+1 if difference<=.5 else 0
        if index>=9 and stable>=4:
            return current
        previous=current
    raise ValueError('滚动衔接画面持续变化，停止以免跳过节点')


def select_rect(screenshot, title):
    import tkinter as tk
    root = tk.Tk()
    root.title(title)
    root.attributes('-fullscreen', True)
    root.attributes('-topmost', True)
    canvas = tk.Canvas(root, highlightthickness=0, cursor='crosshair')
    canvas.pack(fill='both', expand=True)
    photo = ImageTk.PhotoImage(screenshot)
    canvas.create_image(0, 0, anchor='nw', image=photo)
    canvas.create_rectangle(0, 0, screenshot.width, 48, fill='#15263c')
    canvas.create_text(20, 24, text=title+'；拖框选择，松手确认；Esc 取消',
                       fill='white', anchor='w', font=('Microsoft YaHei', 13))
    state = {}
    def down(event):
        state['start'] = (event.x, event.y)
        state['box'] = canvas.create_rectangle(event.x, event.y, event.x, event.y,
                                                outline='#ff592f', width=2)
    def move(event):
        if 'start' in state:
            canvas.coords(state['box'], *state['start'], event.x, event.y)
    def up(event):
        if 'start' not in state:
            return
        x, y = state['start']
        rect = [min(x,event.x), min(y,event.y), abs(x-event.x), abs(y-event.y)]
        if min(rect[2:]) >= 5:
            state['rect'] = rect
            root.destroy()
    canvas.bind('<ButtonPress-1>', down)
    canvas.bind('<B1-Motion>', move)
    canvas.bind('<ButtonRelease-1>', up)
    root.bind('<Escape>', lambda _: root.destroy())
    root.mainloop()
    if 'rect' not in state:
        raise KeyboardInterrupt('取消校准')
    return state['rect']


def calibrate(args):
    from win_input import Windows
    win = Windows()
    print('请在主屏显示企业微信通讯录，滚到最顶部，同时保留一个收起部门和一个展开部门。5 秒后截图。', flush=True)
    time.sleep(5)
    screen = ImageGrab.grab()
    tree = select_rect(screen, '1/5 框选组织树内容：从首行上方约半行开始，排除标题和滚动条')
    recording = select_rect(screen, '2/5 框选录屏区域：同时包含组织树和完整右侧详情')
    closed = select_rect(screen, '3/5 紧框一个向右的灰色收起箭头：少留空白，不包含文件夹')
    opened = select_rect(screen, '4/5 紧框一个向下的灰色展开箭头：少留空白，不包含文件夹')
    folder_icon = select_rect(screen, '5/5 紧框一个未选中部门的蓝色文件夹图标：不包含箭头和文字')
    hwnd, window_rect = win.target_at(tree[0]+tree[2]//2, tree[1]+tree[3]//2)
    for rect in (tree, recording):
        x, y, w, h = rect
        if not (window_rect[0] <= x < x+w <= window_rect[2]
                and window_rect[1] <= y < y+h <= window_rect[3]):
            raise ValueError('树和录屏区域必须都位于企业微信窗口内')
    x, y, w, h = recording
    if not (x <= tree[0] and y <= tree[1] and x+w >= tree[0]+tree[2] and y+h >= tree[1]+tree[3]):
        raise ValueError('录屏范围必须包含完整树区域')
    for name, rect in [('closed.png', closed), ('opened.png', opened),('folder.png',folder_icon)]:
        x, y, w, h = rect
        screen.crop((x, y, x+w, y+h)).save(BASE/name)
    config = dict(tree=tree, recording=recording, hwnd=hwnd, window_rect=window_rect,
                  spacing=args.spacing, dwell=2.5, expand_wait=1.5, scroll_wait=1.2,
                  fps=4, ffmpeg='ffmpeg', max_nodes=10000, max_minutes=180)
    (BASE/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    x, y, w, h = tree
    crop = cv2.cvtColor(np.array(screen.crop((x,y,x+w,y+h))), cv2.COLOR_RGB2BGR)
    annotate(crop, config, BASE/'calibration-preview.png')
    print('校准已保存。请先核对 calibration-preview.png：每行应有一个框，部门箭头位置正确。')


def annotate(image, config, output):
    canvas = image.copy()
    templates = [(read_image(BASE/'closed.png'), (0,0,255)),
                 (read_image(BASE/'opened.png'), (0,180,0))]
    found = rows(image, config['spacing'])
    for i, row in enumerate(found):
        cv2.rectangle(canvas, (0,row.top), (image.shape[1]-1,row.bottom), (210,110,0), 1)
        cv2.putText(canvas, str(i+1), (image.shape[1]-35,row.y), cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,180),1)
        for template, color in templates:
            match = arrow(image,row,template)
            if match:
                cv2.circle(canvas, (match[0],match[1]), 5,color,1)
    save_image(output, canvas)
    return len(found)


def run(args):
    from win_input import Windows
    from recorder import Recorder
    config = json.loads((BASE/'config.json').read_text(encoding='utf-8'))
    resume = Path(args.resume).resolve() if args.resume else None
    resume_y = None
    previous_image = None
    last_kind = None
    resume_scroll = None
    if resume:
        old_config = json.loads((resume/'config.json').read_text(encoding='utf-8'))
        for key in ('tree','recording','hwnd','window_rect','spacing'):
            if old_config[key] != config[key]:
                raise ValueError('续录时窗口及录屏布局必须与上一段一致')
        old_events = [json.loads(s) for s in (resume/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        old_summary = json.loads((resume/'summary.json').read_text(encoding='utf-8'))
        visits = [e for e in old_events if e['kind']=='visited']
        if not visits or old_summary['status'] not in ('limit_reached','stopped','blocked'):
            raise ValueError('没有可续录的成功节点，或上一段状态不支持续录')
        last = visits[-1]
        last_kind = last['kind_hint']
        recovery_reason=any(reason in old_summary['reason'] for reason in ('滚动衔接','姓名锚点不匹配','相邻节点间距','未确认目标行被选中','企业微信失去前台焦点'))
        if any(e['kind']=='scroll' and e['elapsed']>last['elapsed'] for e in old_events) and not (args.resume_after_scroll and old_summary['status']=='blocked' and recovery_reason):
            raise ValueError('上一段在最后节点后发生了滚动，不能按旧画面续录')
        previous_image = read_image(resume/f"{last['index']:06d}.png")
        rx,ry,_,_ = old_config['recording']
        tx,ty,tw,th = old_config['tree']
        previous_image = previous_image[ty-ry:ty-ry+th,tx-rx:tx-rx+tw]
        resume_y = last['y']
    if args.dwell is not None:
        config['dwell'] = args.dwell
    if not 1 <= config['dwell'] <= 30 or not 16 <= config['spacing'] <= 100:
        raise ValueError('停留时间要求 1–30 秒，行距要求 16–100 像素')
    limit = args.limit if args.limit is not None else config['max_nodes']
    if limit <= 0:
        raise ValueError('节点上限必须大于 0')
    win = Windows()
    closed = read_image(BASE/'closed.png')
    opened = read_image(BASE/'opened.png')
    folder_template = read_image(BASE/'folder.png')
    footer_template = read_image(BASE/'footer.png') if (BASE/'footer.png').exists() else None
    if min(float(closed.std()), float(opened.std())) < 10:
        raise ValueError('箭头模板缺少有效图像，请重新校准')
    folder = Path(args.output_dir).resolve() if args.output_dir else BASE/'runs'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    folder.mkdir(parents=True)
    # 每段保存实际识别模板，后续校准改变也不影响该段审计依据。
    for name, template in [('closed.png',closed),('opened.png',opened),('folder.png',folder_template)]:
        save_image(folder/name,template)
    if footer_template is not None:
        save_image(folder/'footer.png',footer_template)
    (folder/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    config['resume_from'] = str(resume) if resume else None
    config['snapshots'] = bool(args.snapshots)
    (folder/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    print('5 秒后开始。请切到企业微信，'+('保持上段结束画面' if resume else '树滚至顶部')+'；F8 暂停/继续，Esc 停止。', flush=True)
    time.sleep(5)
    win.checkpoint(config)
    if previous_image is not None:
        current = grab(config['tree'])
        difference = float(np.mean(np.abs(current.astype(float)-previous_image.astype(float)))) if current.shape==previous_image.shape else 255
        if difference > 1.0:
            if not args.resume_after_scroll or old_summary['status']!='blocked' or not recovery_reason:
                raise ValueError(f'当前树与上段末帧不同（平均像素差 {difference:.2f}），拒绝盲目续录')
            half=config['spacing']//2
            top=max(0,resume_y-half)
            anchor=previous_image[top:min(len(previous_image),resume_y+half),:-12]
            new_y=locate_anchor(current,anchor,resume_y,center_offset=resume_y-top,
                                allow_avatar_change=last_kind=='person_or_unmatched')
            # 同时核对上一行上下文，不能只凭一个选中姓名改变断点。
            before_top=resume_y-config['spacing']-half
            after_top=new_y-config['spacing']-half
            if min(before_top,after_top)<0:
                raise ValueError('恢复滚动缺少上一行上下文')
            prior=previous_image[before_top:before_top+2*half,:-12]
            actual=current[after_top:after_top+2*half,:-12]
            context_score=float(cv2.matchTemplate(actual,prior,cv2.TM_CCOEFF_NORMED)[0,0])
            if context_score<.96:
                raise ValueError('滚动恢复的上一行上下文不匹配')
            resume_scroll=dict(before=resume_y,after=new_y,context_score=context_score)
            resume_y=new_y
            save_image(folder/'resume-after-scroll.png',grab(config['recording']))
    recorder = None
    status, reason, count = 'interrupted', '', 0
    start = time.monotonic()
    last_y = resume_y if resume_y is not None else -config['spacing']
    from vision import Row
    last_row = Row(last_y,last_y-config['spacing']//2,last_y+config['spacing']//2) if resume else None
    unmoved = 0
    log = (folder/'events.jsonl').open('w',encoding='utf-8')
    def event(kind, **fields):
        log.write(json.dumps(dict(kind=kind, elapsed=round(time.monotonic()-start,3),
                                 **fields),ensure_ascii=False)+'\n')
        log.flush()
    try:
        recorder = Recorder(folder,config['recording'],config['ffmpeg'],config['fps'])
        event('recording_started')
        if resume:
            event('resumed',source=str(resume),y=last_y)
            if resume_scroll:
                event('resumed_after_scroll',**resume_scroll)
        while count < limit:
            win.checkpoint(config)
            recorder.check()
            if time.monotonic()-start > config['max_minutes']*60:
                status, reason = 'limit_reached', '达到运行时间上限'
                break
            image = grab(config['tree'])
            candidates = [r for r in rows(image,config['spacing'],footer_template) if r.y > last_y+config['spacing']*.6]
            if candidates:
                row = candidates[0]
                if last_row is not None and row.y-last_y > config['spacing']*1.55:
                    raise RuntimeError('相邻节点间距超过一行，可能漏识别节点，停止核对')
                c, o = arrow(image,row,closed), arrow(image,row,opened)
                if folder_template is not None and arrow(image,row,folder_template,.88) and not (c or o):
                    raise RuntimeError('检测到文件夹但无法确定箭头方向，停止以免当成人员而漏掉子部门')
                if c and o:
                    raise RuntimeError('同一行同时匹配展开和收起箭头，请重新校准')
                tx, ty, tw, th = config['tree']
                # 点击树行右侧空白，避开最左侧箭头；若客户端不支持行空白选中，试跑必须先发现。
                win.click(tx+tw-24,ty+row.y,config)
                win.wait(config['dwell'],config,recorder)
                changed = grab(config['tree'])
                selection_deadline = time.monotonic()+10
                delayed_selection = False
                while not selected(changed,row) and time.monotonic()<selection_deadline:
                    # 只等待界面更新，不重发点击（部门整行重点击会收起）。
                    delayed_selection = True
                    win.wait(.25,config,recorder)
                    changed = grab(config['tree'])
                if not selected(changed,row):
                    save_image(folder/'selection-failed.png',grab(config['recording']))
                    raise RuntimeError('未确认目标行被选中，请检查树右侧空白是否可点击、遮挡或主题变化')
                if delayed_selection:
                    event('selection_delayed',y=row.y)
                    win.wait(config['dwell'],config,recorder)
                    changed = grab(config['tree'])
                if c or o:
                    # 企业微信部门整行点击会切换展开状态。先选中并录制部门，
                    # 再检查实际状态，只在仍收起时展开，后面不再点击整行。
                    now_closed, now_open = arrow(changed,row,closed), arrow(changed,row,opened)
                    if now_closed and not now_open:
                        win.click(tx+now_closed[0],ty+now_closed[1],config)
                        win.wait(config['expand_wait'],config,recorder)
                        changed = grab(config['tree'])
                        event('expanded', y=row.y)
                    if arrow(changed,row,closed) or not arrow(changed,row,opened):
                        raise RuntimeError('未确认部门最终保持展开；停止，避免漏掉子节点')
                count += 1
                last_y, last_row = row.y, row
                last_kind = 'department' if c or o else 'person_or_unmatched'
                event('visited', index=count, y=row.y, kind_hint='department' if c or o else 'person_or_unmatched')
                if args.snapshots:
                    save_image(folder/f'{count:06d}.png',grab(config['recording']))
                print(f'已浏览 {count} 行', flush=True)
                continue
            if last_row is None:
                raise RuntimeError('未检测到完整节点行，请检查树区域和行距')
            anchor = image[last_row.top:last_row.bottom, :image.shape[1]-12].copy()
            tx, ty, tw, th = config['tree']
            win.scroll(tx+tw//2,ty+th//2,config)
            win.wait(config['scroll_wait'],config,recorder)
            after = settled_tree(lambda: grab(config['tree']),
                                 lambda seconds: win.wait(seconds,config,recorder))
            new_y = locate_anchor(after,anchor,last_y,center_offset=last_y-last_row.top,
                                  allow_avatar_change=last_kind=='person_or_unmatched')
            if abs(new_y-last_y) <= 3:
                unmoved += 1
                if unmoved >= 3:
                    status = 'end_candidate'
                    reason = '连续三次滚动未移动：可能到达末尾，需人工核对；不等于通讯录完整性证明'
                    break
            else:
                unmoved = 0
            event('scroll', before=last_y, after=new_y)
            last_y = new_y
            from vision import Row
            last_row = Row(new_y,max(0,new_y-config['spacing']//2),min(len(after),new_y+config['spacing']//2))
        else:
            status, reason = 'limit_reached', '达到试跑/节点上限，尚未证明遍历完毕'
    except KeyboardInterrupt:
        status, reason = 'stopped', '用户停止'
    except Exception as error:
        status, reason = 'blocked', str(error)
    finally:
        if recorder:
            try:
                recorder.close()
            except Exception as error:
                status, reason = 'recording_error', str(error)
        event('finished',status=status,reason=reason,visited_rows=count)
        log.close()
        (folder/'summary.json').write_text(json.dumps(dict(status=status,reason=reason,
            visited_rows=count,coverage_verified=False,elapsed_seconds=round(time.monotonic()-start,2)),
            ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'{reason}\n结果：{folder}',flush=True)
    return 1 if status in ('blocked','recording_error') else 0


def main():
    parser = argparse.ArgumentParser(description='企业微信组织树逐行浏览与本地录屏，无模型、无 OCR。')
    sub = parser.add_subparsers(dest='command',required=True)
    calibration = sub.add_parser('calibrate',help='五步框选校准，生成识别预览')
    calibration.add_argument('--spacing',type=int,default=36,help='相邻行中心距离（实际像素），截图示例为 36')
    execute = sub.add_parser('run',help='开始自动浏览并录屏')
    execute.add_argument('--limit',type=int,help='最多浏览行数，首次建议 10')
    execute.add_argument('--dwell',type=float,help='每行点击后停留秒数，默认 2.5')
    execute.add_argument('--snapshots',action='store_true',help='额外保存每行录屏区域的无损 PNG')
    execute.add_argument('--resume',help='从上一段成功末帧续录；要求上一段保存了 PNG 且树画面未变')
    execute.add_argument('--resume-after-scroll',action='store_true',help='仅恢复滚动衔接中断：验证原选中人员/节点及上一行上下文后接续')
    execute.add_argument('--output-dir',help='指定不存在的新输出目录，供分段录制器使用')
    args = parser.parse_args()
    try:
        return calibrate(args) if args.command == 'calibrate' else run(args)
    except (Exception,KeyboardInterrupt) as error:
        print(f'未开始/已停止：{error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
