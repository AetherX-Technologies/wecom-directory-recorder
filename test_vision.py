import unittest
import cv2
import numpy as np
from vision import Row, rows, arrow, locate_anchor, selected


class VisionTests(unittest.TestCase):
    def fixture(self):
        image = np.full((180,220,3),245,np.uint8)
        for y in [24,60,96,132,168]:
            cv2.rectangle(image,(24,y-9),(41,y+9),(220,140,50),-1)
            cv2.putText(image,str(y),(50,y+5),cv2.FONT_HERSHEY_SIMPLEX,.45,(30,30,30),1)
        return image

    def test_rows_exclude_partial_bottom(self):
        self.assertEqual([r.y for r in rows(self.fixture()[:173])],[24,60,96,132])

    def test_last_row_not_lost_when_only_blank_margin_clipped(self):
        self.assertEqual([r.y for r in rows(self.fixture())],[24,60,96,132,168])

    def test_blank(self):
        self.assertEqual(rows(np.full((180,220,3),245,np.uint8)),[])

    def test_selected_row_is_one_row(self):
        image=self.fixture()
        image[44:77,:]=(250,140,50)
        self.assertEqual([r.y for r in rows(image)],[24,60,96,132,168])
        self.assertTrue(selected(image,Row(60,42,78)))
        self.assertFalse(selected(image,Row(96,78,114)))

    def test_arrow_orientation(self):
        image=self.fixture()
        cv2.fillPoly(image,[np.array([[8,53],[8,65],[14,59]])],(150,150,150))
        template=image[51:68,6:17].copy()
        self.assertIsNotNone(arrow(image,Row(60,42,78),template))
        self.assertIsNone(arrow(image,Row(96,78,114),template))

    def test_selection_does_not_merge_with_previous_icon(self):
        image=self.fixture()
        image[44:77,:]=(250,140,50)
        image[30:39,24:33]=(30,30,30)
        found=rows(image)
        self.assertEqual(len(found),5)
        self.assertEqual(found[1].y,60)

    def test_long_white_name_does_not_hide_blue_selection(self):
        image=self.fixture()
        image[44:77,:]=(250,140,50)
        image[53:68,15:210]=(255,255,255)
        self.assertTrue(selected(image,Row(60,42,78)))
        self.assertFalse(selected(image,Row(96,78,114)))

    def test_anchor_shift(self):
        image=self.fixture()
        anchor=image[114:150,:208].copy()
        shifted=np.full_like(image,245)
        shifted[:-72]=image[72:]
        self.assertEqual(locate_anchor(shifted,anchor,132),60)

    def test_arrow_inverted_selection_contrast(self):
        image=self.fixture()
        cv2.fillPoly(image,[np.array([[8,53],[8,65],[14,59]])],(150,150,150))
        template=image[51:68,6:17].copy()
        image[42:78]=255-image[42:78]
        self.assertIsNotNone(arrow(image,Row(60,42,78),template))

    def test_anchor_missing_stops(self):
        image=self.fixture()
        with self.assertRaises(ValueError):
            locate_anchor(np.full_like(image,245),image[114:150,:208],132)

    def test_clipped_margin_anchor_retains_true_row_center(self):
        image=self.fixture()
        anchor=image[150:180,:208]
        self.assertEqual(locate_anchor(image,anchor,168,center_offset=18),168)

    def test_duplicate_anchor_stops(self):
        image=self.fixture()
        anchor=image[114:150,:208].copy()
        image[6:42,:208]=anchor
        with self.assertRaises(ValueError):
            locate_anchor(image,anchor,132)

    def test_backward_scroll_stops(self):
        image=self.fixture()
        with self.assertRaises(ValueError):
            locate_anchor(image,image[114:150,:208],60)

    def avatar_fixture(self, name='ALICE'):
        row=np.full((36,208,3),(250,140,50),np.uint8)
        row[8:28,24:44]=(210,210,210)
        cv2.putText(row,name,(50,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        changed=row.copy()
        changed[8:28,24:44]=(20,30,40)
        image=np.full((180,208,3),245,np.uint8)
        image[42:78]=changed
        return row,image

    def test_avatar_change_requires_explicit_opt_in(self):
        anchor,image=self.avatar_fixture()
        with self.assertRaises(ValueError):
            locate_anchor(image,anchor,132)
        self.assertEqual(locate_anchor(image,anchor,132,allow_avatar_change=True),60)

    def test_avatar_fallback_rejects_other_name(self):
        anchor,_=self.avatar_fixture()
        _,image=self.avatar_fixture('BOB')
        with self.assertRaises(ValueError):
            locate_anchor(image,anchor,132,allow_avatar_change=True)

    def test_avatar_fallback_rejects_duplicate_and_backward(self):
        anchor,image=self.avatar_fixture()
        with self.assertRaises(ValueError):
            locate_anchor(image,anchor,24,allow_avatar_change=True)
        image[114:150]=image[42:78]
        with self.assertRaises(ValueError):
            locate_anchor(image,anchor,168,allow_avatar_change=True)


if __name__=='__main__':
    unittest.main()
