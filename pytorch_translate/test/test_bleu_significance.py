#!/usr/bin/env python3

import tempfile
import unittest

import sacrebleu
from pytorch_translate import bleu_significance


class BleuSignificanceTest(unittest.TestCase):
    def _setup_files(self, reference_file, baseline_file, new_file):
        reference_file.write(
            r"""Lorem ipsum dolor sit amet , consectetur adipiscing elit , sed do eiusmod tempor incididunt ut labore et dolore magna aliqua .
Ut enim ad minim veniam , quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat .
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur .
Excepteur sint occaecat cupidatat non proident , sunt in culpa qui officia deserunt mollit anim id est laborum .
Short sentence .
Complete gibberish"""
        )
        reference_file.flush()
        baseline_file.write(
            r"""Lorem ipsum dolor sit AAAA , consectetur adipiscing elit , sed DD eiusmod tempor incididunt ut labore et DDDDDD magna aliqua .
Ut enim ad MMMMM veniam , quis nostrud exercitation ullamco laboris NNNN ut aliquip EE ea commodo consequat .
Duis aute irure dolor in reprehenderit in voluptate VVVVV EEEE cillum dolore eu FFFFFF nulla pariatur .
Excepteur sint occaecat cupidatat non proident , SSSS in culpa QQQ officia deserunt mollit AAAA id EEE laborum .
Short phrase .
O_o"""
        )
        baseline_file.flush()
        new_file.write(
            r"""Lorem IIIII dolor sit amet , consectetur adipiscing EEEE , sed do eiusmod tempor incididunt UU labore et dolore magna aliqua .
Ut enim ad MMMMM veniam , QQQQ nostrud exercitation ullamco laboris nisi ut aliquip ex EE commodo consequat .
DDDD aute irure dolor in reprehenderit in voluptate velit esse cillum dolore EE fugiat NNNNN pariatur .
Excepteur SSSS occaecat cupidatat NNN proident , SSSS in culpa qui officia deserunt mollit anim id EEE laborum .
Short sentence
Hodor hodor hodor hodor hodor hodor hodor"""
        )
        new_file.flush()

    def test_paired_bootstrap_resample_from_files(self):
        with tempfile.NamedTemporaryFile(
            mode="w+"
        ) as reference_file, tempfile.NamedTemporaryFile(
            mode="w+"
        ) as baseline_file, tempfile.NamedTemporaryFile(
            mode="w+"
        ) as new_file:
            self._setup_files(
                reference_file=reference_file,
                baseline_file=baseline_file,
                new_file=new_file,
            )
            output = bleu_significance.paired_bootstrap_resample_from_files(
                reference_file=reference_file.name,
                baseline_file=baseline_file.name,
                new_file=new_file.name,
            )

            # Sanity checks that our logic calculating BLEU score by combining
            # sufficient statistics is the same as SacreBLEU's.
            reference_file.seek(0)
            reference_lines = [line for line in reference_file]
            baseline_file.seek(0)
            baseline_lines = [line for line in baseline_file]
            new_file.seek(0)
            new_lines = [line for line in new_file]

            self.assertEqual(
                output.baseline_bleu,
                sacrebleu.corpus_bleu(
                    sys_stream=baseline_lines,
                    ref_streams=[reference_lines],
                    lowercase=False,
                    tokenize="none",
                    use_effective_order=False,
                ),
            )
            self.assertEqual(
                output.new_bleu,
                sacrebleu.corpus_bleu(
                    sys_stream=new_lines,
                    ref_streams=[reference_lines],
                    lowercase=False,
                    tokenize="none",
                    use_effective_order=False,
                ),
            )
