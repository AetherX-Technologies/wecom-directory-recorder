import ctypes
from ctypes import wintypes
import unittest
from unittest.mock import Mock, patch
from win_input import Windows


class FocusTests(unittest.TestCase):
    def run_case(self, handles, window_class='Ghost', escape=False):
        window = Windows.__new__(Windows)
        window.u = Mock()
        window.paused = window.was_f8 = False
        window.geometry = Mock(return_value=[0, 0, 10, 10])
        window.u.GetForegroundWindow.side_effect = handles
        window.u.GetClassNameW.side_effect = lambda handle, buf, count: setattr(buf, 'value', window_class)
        keys = iter([0, 0, 0x8000 if escape else 0])
        window.u.GetAsyncKeyState.side_effect = lambda key: next(keys, 0)
        def cursor(pointer):
            value = ctypes.cast(pointer, ctypes.POINTER(wintypes.POINT)).contents
            value.x = value.y = 100
        window.u.GetCursorPos.side_effect = cursor
        config = dict(hwnd=1, window_rect=[0, 0, 10, 10])
        clock = [0.0]
        def sleep(seconds):
            clock[0] += seconds
        with patch('win_input.time.monotonic', side_effect=lambda: clock[0]), patch('win_input.time.sleep', side_effect=sleep), patch('win_input.Path.exists', return_value=False):
            result = window.checkpoint(config)
        window.u.mouse_event.assert_not_called()
        return result

    def test_ghost_waits_without_input_until_target_returns(self):
        self.assertAlmostEqual(self.run_case([2, 2, 1]), .2)

    def test_other_window_is_not_tolerated(self):
        with self.assertRaises(RuntimeError):
            self.run_case([2], 'OtherApp')

    def test_ghost_timeout_stops(self):
        with self.assertRaises(RuntimeError):
            self.run_case([2]*400)

    def test_escape_during_ghost_wait_stops(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_case([2], escape=True)


if __name__ == '__main__':
    unittest.main()
