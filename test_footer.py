import unittest
import cv2
import numpy as np
from vision import rows


class FooterTests(unittest.TestCase):
    def fixture(self):
        image = np.full((180,220,3),245,np.uint8)
        for y in [24,60,96,132]:
            cv2.rectangle(image,(24,y-9),(41,y+9),(220,140,50),-1)
        cv2.putText(image,'6547',(89,174),cv2.FONT_HERSHEY_SIMPLEX,.45,(150,150,150),1)
        return image, image[159:179,86:130].copy()

    def test_only_confirmed_footer_is_excluded(self):
        image, footer = self.fixture()
        self.assertEqual(len(rows(image)),5)
        self.assertEqual(len(rows(image,footer_template=footer)),4)

    def test_mismatched_footer_is_not_excluded(self):
        image, footer = self.fixture()
        footer = np.flip(footer,axis=1).copy()
        self.assertEqual(len(rows(image,footer_template=footer)),5)

    def test_selected_bottom_person_is_preserved(self):
        image, footer = self.fixture()
        image[151:177,:] = (250,140,50)
        self.assertEqual(len(rows(image,footer_template=footer)),5)


if __name__ == '__main__':
    unittest.main()
