#!/usr/bin/env python3

MAX_SPECIAL_TOKENS = 100

# Number of Byte indices is always fixed at 256 (0-255). The additional 5 indices
# correpsond to the special tokens for byte numberization including
# padding, start and end of word, start and end of sentence. These are
# separate from the special tokens in the dict and match up with the indices
# used by pre-trained ELMo.
NUM_BYTE_INDICES = 261

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
MASK_ID = 4

DIALECT_CODES = {
    "af_ZA": 8,
    "am_ET": 9,
    "ar_AR": 10,
    "be_BY": 11,
    "bg_BG": 12,
    "bn_IN": 13,
    "br_FR": 14,
    "bs_BA": 15,
    "ca_ES": 16,
    "cs_CZ": 17,
    "cy_GB": 18,
    "da_DK": 19,
    "de_DE": 20,
    "el_GR": 21,
    "en_XX": 22,
    "es_XX": 23,
    "et_EE": 24,
    "eu_ES": 25,
    "fa_IR": 26,
    "fi_FI": 27,
    "fr_XX": 28,
    "gu_IN": 29,
    "ha_NG": 30,
    "he_IL": 31,
    "hi_IN": 32,
    "hr_HR": 33,
    "hu_HU": 34,
    "id_ID": 35,
    "it_IT": 36,
    "ja_XX": 37,
    "km_KH": 38,
    "kn_IN": 39,
    "ko_KR": 40,
    "lt_LT": 41,
    "lv_LV": 42,
    "mk_MK": 43,
    "ml_IN": 44,
    "mn_MN": 45,
    "mr_IN": 46,
    "ms_MY": 47,
    "my_MM": 48,
    "ne_NP": 49,
    "nl_XX": 50,
    "no_XX": 51,
    "pa_IN": 52,
    "pl_PL": 53,
    "ps_AF": 54,
    "pt_XX": 55,
    "ro_RO": 56,
    "ru_RU": 57,
    "si_LK": 57,
    "sk_SK": 58,
    "sl_SI": 59,
    "so_SO": 60,
    "sq_AL": 61,
    "sr_RS": 62,
    "sv_SE": 63,
    "sw_KE": 64,
    "ta_IN": 65,
    "te_IN": 66,
    "th_TH": 67,
    "tl_XX": 68,
    "tr_TR": 69,
    "uk_UA": 70,
    "ur_PK": 71,
    "vi_VN": 72,
    "xh_ZA": 73,
    "zh_CN": 74,
    "zh_TW": 75,
    "zu_ZA": 76,
}
