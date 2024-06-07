import unittest
from TextGen.utils import DataLoader


class LossTest(unittest.TestCase):
    def test_exception_wrong_name(self):
        with self.assertRaises(FileNotFoundError):
            DataLoader.load('TextGen/tests/utils/data_loader/test_wrong.txt')

    def test_load_data(self):
        data = DataLoader.load(
            'TextGen/tests/utils/data_loader/krol_karol.txt')
        self.assertEqual(
            data.raw,
            'Król Karol kupił królowej Karolinie korale koloru koralowego.')

    def test_load_multiline_data(self):
        data = DataLoader.load(
            'TextGen/tests/utils/data_loader/krol_karol_long.txt')
        self.assertEqual(
            data.raw,
            'Król Karol kupił królowej Karolinie korale koloru koralowego,\nKrólowa Karolina kolory korali królowi Karolowi kontrolowała,\nKiedy król Karol kolekcję korali królowej Karolinie komplementował.')

    def test_word_tokenize(self):
        data = DataLoader.load(
            'TextGen/tests/utils/data_loader/krol_karol.txt')
        self.assertEqual(data.word_tokenize(),
                         ['Król',
                          'Karol',
                          'kupił',
                          'królowej',
                          'Karolinie',
                          'korale',
                          'koloru',
                          'koralowego',
                          '.'])

    def test_special_characters(self):
        data = DataLoader.load(
            'TextGen/tests/utils/data_loader/special_characters.txt')
        self.assertEqual(data.word_tokenize(),
                         ['.',
                          '!',
                          '?',
                          ',',
                          ';',
                          ':',
                          '@',
                          '#',
                          '$',
                          '%',
                          '&',
                          '(',
                          ')',
                          '[',
                          ']',
                          '{',
                          '}',
                          '<',
                          '>',
                          "''",
                          "'",
                          '...',
                          '--'])
