import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
import numpy as np

from startup import match_entry, validate_profile, open_window, navigate, prepare


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(window_rect=[10,20,410,320],tree=[110,80,100,200],
                           recording=[110,40,280,250],spacing=36,hwnd=1)
        self.profile = dict(executable='WXWork.exe',layout={k:self.config[k] for k in
                            ('window_rect','tree','recording','spacing')},
                            steps=[dict(template='contact.png',region=[0,0,100,200])],
                            ready=dict(template='ready.png',region=[100,0,100,40]))
        self.template = np.random.default_rng(12).integers(0,256,(12,16,3),dtype=np.uint8)

    def test_unique_entry_coordinates(self):
        image=np.full((100,120,3),230,np.uint8)
        image[30:42,20:36]=self.template
        self.assertEqual(match_entry(image,self.template),(28,36))

    def test_duplicate_entry_rejected(self):
        image=np.full((100,120,3),230,np.uint8)
        image[30:42,20:36]=self.template
        image[65:77,70:86]=self.template
        with self.assertRaisesRegex(ValueError,'不唯一'):
            match_entry(image,self.template)

    def test_missing_entry_and_blank_template_rejected(self):
        blank=np.full((100,120,3),230,np.uint8)
        with self.assertRaises(ValueError):
            match_entry(blank,self.template)
        with self.assertRaisesRegex(ValueError,'特征'):
            match_entry(blank,blank[:12,:16])

    def test_layout_change_requires_setup(self):
        config=copy.deepcopy(self.config)
        config['tree'][0]+=10
        with self.assertRaisesRegex(ValueError,'校准已改变'):
            validate_profile(self.profile,config)

    def test_invalid_region_rejected(self):
        self.profile['steps'][0]['region']=[350,0,100,200]
        with self.assertRaisesRegex(ValueError,'超出窗口'):
            validate_profile(self.profile,self.config)

    def test_new_handle_can_replace_old_after_restart(self):
        validate_profile(self.profile,dict(self.config,hwnd=999))
        desktop=Mock()
        desktop.find.return_value=99
        launch=Mock()
        with patch('startup.time.sleep'):
            runtime=open_window(desktop,self.profile,self.config,launch)
        launch.assert_not_called()
        self.assertEqual(runtime['hwnd'],99)
        self.assertEqual(self.config['hwnd'],1)
        desktop.restore.assert_called_once_with(99,self.config['window_rect'])

    def test_absent_window_launches_once_and_refreshes(self):
        desktop=Mock()
        desktop.find.side_effect=[None,None,99]
        desktop.u.GetAsyncKeyState.return_value=0
        launch=Mock()
        with patch('startup.time.sleep'),patch('startup.Path.exists',return_value=False):
            runtime=open_window(desktop,self.profile,self.config,launch)
        launch.assert_called_once_with('WXWork.exe')
        self.assertEqual(runtime['hwnd'],99)

    def test_login_timeout_does_not_restore_or_click(self):
        desktop=Mock()
        desktop.find.return_value=None
        desktop.u.GetAsyncKeyState.return_value=0
        launch=Mock()
        with patch('startup.time.sleep'),patch('startup.Path.exists',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'登录'):
                open_window(desktop,self.profile,self.config,launch)
        desktop.restore.assert_not_called()
        desktop.click.assert_not_called()
        launch.assert_called_once()

    def test_unrecognized_entry_does_not_click(self):
        desktop=Mock()
        capture=Mock(return_value=np.zeros((200,100,3),np.uint8))
        with patch('startup.read_image',return_value=self.template),patch('startup.time.monotonic',side_effect=[0,16]):
            with self.assertRaises(ValueError):
                navigate(desktop,self.profile,self.config,root=Path('.'),capture=capture)
        desktop.click.assert_not_called()

    def test_failed_navigation_does_not_rewrite_calibration(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            executable=root/'WXWork.exe'
            executable.touch()
            self.profile['executable']=str(executable)
            original=json.dumps(self.config).encode()
            (root/'config.json').write_bytes(original)
            (root/'startup-profile.json').write_text(json.dumps(self.profile),encoding='utf-8')
            with patch('startup.BASE',root),patch('startup.Desktop'),patch('startup.open_window',return_value=dict(self.config,hwnd=99)),patch('startup.navigate',side_effect=ValueError('页面不匹配')):
                with self.assertRaises(ValueError):
                    prepare()
            self.assertEqual((root/'config.json').read_bytes(),original)
            self.assertFalse((root/'config.prepare.tmp').exists())


if __name__ == '__main__':
    unittest.main()
