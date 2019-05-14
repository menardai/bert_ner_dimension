import unittest

from model import DimensionBertNer


class TestDimension(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ This is called before tests in an individual class are run. """
        #cls.model = DimensionBertNer('../models/dimension_ner_bert.pt')
        cls.model = DimensionBertNer('../models/dimension_ner_bert_last_epoch.pt')
        #cls.model = DimensionBertNer('../models/dimension_ner_bert_best.pt')

    def setUp(self):
        """ This is called immediately before calling the test method. """
        pass

    def test_WxH(self):
        """ 640x480 """

        # standard
        dim = self.model.predict(["I would like to resize the previous image at 1024x768"])
        self.assertEqual(dim, [{'W': 1024, 'H': 768}])

        # too long (max token is more than 24)
        dim = self.model.predict(["I would like to resize the previous image at 1024 x 768, "
                                  "but if you can't just tell me and I will do like you do!"])
        self.assertEqual(dim, [{'W': 1024, 'H': 768}])

        # in the middle
        dim = self.model.predict(["I want a dimension of 1024x768 for this image."])
        self.assertEqual(dim, [{'W': 1024, 'H': 768}])

        # at the beginning
        dim = self.model.predict(["1024x768 is good."])
        self.assertEqual(dim, [{'W': 1024, 'H': 768}])

    def test_WandH(self):
        """ 640 and 480 """

        # standard
        dim = self.model.predict(["Resize to 1024 and 768"])
        self.assertEqual(dim, [{'W': 1024, 'H': 768}])

    def test_optimization(self):
        # "(number,number)" --> "number x number"
        dim = self.model.predict(["Resize to (1024, 768)."])
        self.assertEqual(dim, [{'W': 1024, 'H': 768}])

        # "(number x number)" --> "number x number"
        dim = self.model.predict(["Resize to (1024x768)."])
        self.assertEqual(dim, [{'W': 1024, 'H': 768}])

        # "number/number" --> "number x number"
        dim = self.model.predict(["Resize to 1024:768"])
        self.assertEqual(dim, [{'W': 1024, 'H': 768}])

        # "number/number" --> "number x number"
        dim = self.model.predict(["Resize to (1024:768)"])
        self.assertEqual(dim, [{'W': 1024, 'H': 768}])

        # "number/number" --> "number x number"
        dim = self.model.predict(["Resize to 1024/768"])
        self.assertEqual(dim, [{'W': 1024, 'H': 768}])

        # ("number/number") --> "number x number"
        dim = self.model.predict(["Resize to ( 1024 / 768 )"])
        self.assertEqual(dim, [{'W': 1024, 'H': 768}])

    def test_WandH_explicit(self):
        """ width of 640 and height of 480 """
        dim = self.model.predict(["Resize to a width of 1024 and height of 768"])
        self.assertEqual(dim, [{'W': 1024, 'H': 768}])

        dim = self.model.predict(["width=1024 and height=768"])
        self.assertEqual(dim, [{'W': 1024, 'H': 768}])

    def test_HandW_explicit(self):
        """" height of 480 and width of 640 """
        dim = self.model.predict(["Resize to a height of 768 and width of 1024"])
        self.assertEqual(dim, [{'W': 1024, 'H': 768}])

    def test_W_only_explicit(self):
        """" width=640 """
        dim = self.model.predict(["width=1024"])
        self.assertEqual(dim, [{'W': 1024}])

        dim = self.model.predict(["A width of 1024"])
        self.assertEqual(dim, [{'W': 1024}])

    def test_H_only_explicit(self):
        """" height=480 """
        dim = self.model.predict(["height=768 please"])
        self.assertEqual(dim, [{'H': 768}])

    def test_wrong_W(self):
        """" Not 2, width is 640 """
        dim = self.model.predict(["You got it wrong it's not 2, the width is 1024"])
        self.assertEqual(dim, [{'W': 1024}])

    def test_wrong_H(self):
        """" Not 2, height is 480 """
        dim = self.model.predict(["Bad, the height should be 768 and not 2"])
        self.assertEqual(dim, [{'H': 768}])

    def test_wrong_H_my_bad(self):
        """" not the width, 480 is the height """
        dim = self.model.predict(["This is not the width, 768 is the height"])
        self.assertEqual(dim, [{'H': 768}])


if __name__ == '__main__':
    unittest.main()
