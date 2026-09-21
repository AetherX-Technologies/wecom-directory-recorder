import unittest
import numpy as np
from app import settled_tree


class SettleTests(unittest.TestCase):
    def test_delayed_motion_resets_stability(self):
        values=iter([0,0,0,0,40,70,70,70,70,70,70])
        waits=[]
        result=settled_tree(lambda: np.full((4,4,3),next(values),np.uint8),waits.append)
        self.assertEqual(int(result[0,0,0]),70)
        self.assertEqual(len(waits),10)

    def test_unmoved_still_waits_five_seconds(self):
        waits=[]
        settled_tree(lambda: np.zeros((4,4,3),np.uint8),waits.append)
        self.assertAlmostEqual(sum(waits),5)

    def test_continuous_movement_stops(self):
        values=iter(range(40))
        with self.assertRaises(ValueError):
            settled_tree(lambda: np.full((4,4,3),next(values),np.uint8),lambda _: None)
