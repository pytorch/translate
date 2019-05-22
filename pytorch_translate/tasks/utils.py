#!/usr/bin/env python3

from collections import OrderedDict

from pytorch_translate.data import dictionary as pytorch_translate_dictionary


def load_multilingual_vocabulary(args):
    dicts = OrderedDict()
    vocabulary_list = getattr(args, "vocabulary", [])
    comparison_lang = None
    for vocabulary in vocabulary_list:
        assert (
            ":" in vocabulary
        ), "--vocabulary must be specified in the format lang:path"
        lang, path = vocabulary.split(":")
        dicts[lang] = pytorch_translate_dictionary.Dictionary.load(path)
        if len(dicts) > 1:
            assert dicts[lang].pad() == dicts[comparison_lang].pad()
            assert dicts[lang].eos() == dicts[comparison_lang].eos()
            assert dicts[lang].unk() == dicts[comparison_lang].unk()
        else:
            comparison_lang = lang
        print(f"| [{lang}] dictionary: {len(dicts[lang])} types")

    return dicts
