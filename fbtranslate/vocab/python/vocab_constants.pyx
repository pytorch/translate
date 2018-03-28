from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

cdef extern from \
'language_technology/neural_mt/fbtranslate/vocab/VocabConstants.h' nogil:
    # Static methods
    cdef int getId \
        "facebook::language_technology::neural_mt::VocabConstants::getId"(
            string,
        )
    cdef string getToken \
        "facebook::language_technology::neural_mt::VocabConstants::getToken"(
            string,
        )
    cdef vector[string] SPECIAL_TOKENS_ \
        """
            facebook::language_technology::neural_mt::VocabConstants::
                SPECIAL_TOKENS
        """()

    # Constants
    cdef int MAX_SPECIAL_TOKENS_ \
        """
            facebook::language_technology::neural_mt::VocabConstants::
                MAX_SPECIAL_TOKENS
        """
    cdef int INVALID_ID_ \
        """
            facebook::language_technology::neural_mt::VocabConstants::
                INVALID_ID
        """
    cdef string WORD_VOCAB_TYPE_ \
        """
            facebook::language_technology::neural_mt::VocabConstants::
                WORD_VOCAB_TYPE
        """
    cdef string BPE_VOCAB_TYPE_ \
        """
            facebook::language_technology::neural_mt::VocabConstants::
                BPE_VOCAB_TYPE
        """
    cdef string CHAR_NGRAM_VOCAB_TYPE_ \
        """
            facebook::language_technology::neural_mt::VocabConstants::
                CHAR_NGRAM_VOCAB_TYPE
        """
    cdef string MORFESSOR_VOCAB_TYPE_ \
        """
            facebook::language_technology::neural_mt::VocabConstants::
                MORFESSOR_VOCAB_TYPE
        """
    cdef string WORDPIECE_VOCAB_TYPE_ \
        """
            facebook::language_technology::neural_mt::VocabConstants::
                WORDPIECE_VOCAB_TYPE
        """
    cdef string WORD_VOCAB_TYPE_ \
        """
            facebook::language_technology::neural_mt::VocabConstants::
                WORD_VOCAB_TYPE
        """

# Exposes constants. String constants must be decoded from bytestring (C string)
# to Python string.
MAX_SPECIAL_TOKENS = MAX_SPECIAL_TOKENS_
INVALID_ID = INVALID_ID_

BPE_VOCAB_TYPE = BPE_VOCAB_TYPE_.decode('UTF-8')
CHAR_NGRAM_VOCAB_TYPE = CHAR_NGRAM_VOCAB_TYPE_.decode('UTF-8')
MORFESSOR_VOCAB_TYPE = MORFESSOR_VOCAB_TYPE_.decode('UTF-8')
WORDPIECE_VOCAB_TYPE = WORDPIECE_VOCAB_TYPE_.decode('UTF-8')
WORD_VOCAB_TYPE = WORD_VOCAB_TYPE_.decode('UTF-8')

# Exposes SPECIAL_TOKENS and SPECIAL_TOKENS_TO_ID data structures
def convert_tokens(token_list):
    """
    Converts a token list from bytestring tokens to UTF-8 decoded tokens.
    Inputs:
        token_list: list of bytestrings corresponding to a list of tokens
    Outputs:
        converted_list: list of Python strings corresponding to the token list
        converted_map: dict of Python string token to integer id
    """
    converted_list = []
    converted_map = {}
    token_id = 0
    for token in token_list:
        decoded_token = token.decode('UTF-8')
        converted_list.append(decoded_token)
        converted_map[decoded_token] = token_id
        token_id += 1
    return converted_list, converted_map

SPECIAL_TOKENS, SPECIAL_TOKENS_TO_ID = convert_tokens(SPECIAL_TOKENS_())

"""
Expose any new tokens you added in VocabConstants.cpp below.
If added a token that has a token value but you assigned INVALID_ID to it since
you didn't want to add it to the SPECIAL_TOKENS() list or the
SPECIAL_TOKENS_TO_ID map, just omit providing an ID below.
(See CONTROL_TOKEN for an example, note how there is no CONTROL_ID).
"""
PAD_ID = getId(b'PAD_TOKEN')
PAD_TOKEN = getToken(b'PAD_TOKEN').decode('UTF-8')
GO_ID = getId(b'GO_TOKEN')
GO_TOKEN = getToken(b'GO_TOKEN').decode('UTF-8')
EOS_ID = getId(b'EOS_TOKEN')
EOS_TOKEN = getToken(b'EOS_TOKEN').decode('UTF-8')
UNK_ID = getId(b'UNK_TOKEN')
UNK_TOKEN = getToken(b'UNK_TOKEN').decode('UTF-8')
START_WORD_ID = getId(b'START_WORD_TOKEN')
START_WORD_TOKEN = getToken(b'START_WORD_TOKEN').decode('UTF-8')
END_WORD_ID = getId(b'END_WORD_TOKEN')
END_WORD_TOKEN = getToken(b'END_WORD_TOKEN').decode('UTF-8')
COPY_ID = getId(b'COPY_TOKEN')
COPY_TOKEN = getToken(b'COPY_TOKEN').decode('UTF-8')
UNDEFINED_ID = getId(b'UNDEFINED_TOKEN')
UNDEFINED_TOKEN = getToken(b'UNDEFINED_TOKEN').decode('UTF-8')
CONTROL_TOKEN = getToken(b'CONTROL_TOKEN').decode('UTF-8')

TO_AF_ZA_ID = getId(b'TO_AF_ZA_TOKEN')
TO_AF_ZA_TOKEN = getToken(b'TO_AF_ZA_TOKEN').decode('UTF-8')
TO_AR_AR_ID = getId(b'TO_AR_AR_TOKEN')
TO_AR_AR_TOKEN = getToken(b'TO_AR_AR_TOKEN').decode('UTF-8')
TO_BG_BG_ID = getId(b'TO_BG_BG_TOKEN')
TO_BG_BG_TOKEN = getToken(b'TO_BG_BG_TOKEN').decode('UTF-8')
TO_BN_IN_ID = getId(b'TO_BN_IN_TOKEN')
TO_BN_IN_TOKEN = getToken(b'TO_BN_IN_TOKEN').decode('UTF-8')
TO_BR_FR_ID = getId(b'TO_BR_FR_TOKEN')
TO_BR_FR_TOKEN = getToken(b'TO_BR_FR_TOKEN').decode('UTF-8')
TO_BS_BA_ID = getId(b'TO_BS_BA_TOKEN')
TO_BS_BA_TOKEN = getToken(b'TO_BS_BA_TOKEN').decode('UTF-8')
TO_CA_ES_ID = getId(b'TO_CA_ES_TOKEN')
TO_CA_ES_TOKEN = getToken(b'TO_CA_ES_TOKEN').decode('UTF-8')
TO_CS_CZ_ID = getId(b'TO_CS_CZ_TOKEN')
TO_CS_CZ_TOKEN = getToken(b'TO_CS_CZ_TOKEN').decode('UTF-8')
TO_CY_GB_ID = getId(b'TO_CY_GB_TOKEN')
TO_CY_GB_TOKEN = getToken(b'TO_CY_GB_TOKEN').decode('UTF-8')
TO_DA_DK_ID = getId(b'TO_DA_DK_TOKEN')
TO_DA_DK_TOKEN = getToken(b'TO_DA_DK_TOKEN').decode('UTF-8')
TO_DE_DE_ID = getId(b'TO_DE_DE_TOKEN')
TO_DE_DE_TOKEN = getToken(b'TO_DE_DE_TOKEN').decode('UTF-8')
TO_EL_GR_ID = getId(b'TO_EL_GR_TOKEN')
TO_EL_GR_TOKEN = getToken(b'TO_EL_GR_TOKEN').decode('UTF-8')
TO_EN_XX_ID = getId(b'TO_EN_XX_TOKEN')
TO_EN_XX_TOKEN = getToken(b'TO_EN_XX_TOKEN').decode('UTF-8')
TO_ES_XX_ID = getId(b'TO_ES_XX_TOKEN')
TO_ES_XX_TOKEN = getToken(b'TO_ES_XX_TOKEN').decode('UTF-8')
TO_ET_EE_ID = getId(b'TO_ET_EE_TOKEN')
TO_ET_EE_TOKEN = getToken(b'TO_ET_EE_TOKEN').decode('UTF-8')
TO_EU_ES_ID = getId(b'TO_EU_ES_TOKEN')
TO_EU_ES_TOKEN = getToken(b'TO_EU_ES_TOKEN').decode('UTF-8')
TO_FA_IR_ID = getId(b'TO_FA_IR_TOKEN')
TO_FA_IR_TOKEN = getToken(b'TO_FA_IR_TOKEN').decode('UTF-8')
TO_FI_FI_ID = getId(b'TO_FI_FI_TOKEN')
TO_FI_FI_TOKEN = getToken(b'TO_FI_FI_TOKEN').decode('UTF-8')
TO_FR_XX_ID = getId(b'TO_FR_XX_TOKEN')
TO_FR_XX_TOKEN = getToken(b'TO_FR_XX_TOKEN').decode('UTF-8')
TO_HE_IL_ID = getId(b'TO_HE_IL_TOKEN')
TO_HE_IL_TOKEN = getToken(b'TO_HE_IL_TOKEN').decode('UTF-8')
TO_HI_IN_ID = getId(b'TO_HI_IN_TOKEN')
TO_HI_IN_TOKEN = getToken(b'TO_HI_IN_TOKEN').decode('UTF-8')
TO_HR_HR_ID = getId(b'TO_HR_HR_TOKEN')
TO_HR_HR_TOKEN = getToken(b'TO_HR_HR_TOKEN').decode('UTF-8')
TO_HU_HU_ID = getId(b'TO_HU_HU_TOKEN')
TO_HU_HU_TOKEN = getToken(b'TO_HU_HU_TOKEN').decode('UTF-8')
TO_ID_ID_ID = getId(b'TO_ID_ID_TOKEN')
TO_ID_ID_TOKEN = getToken(b'TO_ID_ID_TOKEN').decode('UTF-8')
TO_IT_IT_ID = getId(b'TO_IT_IT_TOKEN')
TO_IT_IT_TOKEN = getToken(b'TO_IT_IT_TOKEN').decode('UTF-8')
TO_JA_XX_ID = getId(b'TO_JA_XX_TOKEN')
TO_JA_XX_TOKEN = getToken(b'TO_JA_XX_TOKEN').decode('UTF-8')
TO_KO_KR_ID = getId(b'TO_KO_KR_TOKEN')
TO_KO_KR_TOKEN = getToken(b'TO_KO_KR_TOKEN').decode('UTF-8')
TO_LT_LT_ID = getId(b'TO_LT_LT_TOKEN')
TO_LT_LT_TOKEN = getToken(b'TO_LT_LT_TOKEN').decode('UTF-8')
TO_LV_LV_ID = getId(b'TO_LV_LV_TOKEN')
TO_LV_LV_TOKEN = getToken(b'TO_LV_LV_TOKEN').decode('UTF-8')
TO_MK_MK_ID = getId(b'TO_MK_MK_TOKEN')
TO_MK_MK_TOKEN = getToken(b'TO_MK_MK_TOKEN').decode('UTF-8')
TO_MS_MY_ID = getId(b'TO_MS_MY_TOKEN')
TO_MS_MY_TOKEN = getToken(b'TO_MS_MY_TOKEN').decode('UTF-8')
TO_NL_XX_ID = getId(b'TO_NL_XX_TOKEN')
TO_NL_XX_TOKEN = getToken(b'TO_NL_XX_TOKEN').decode('UTF-8')
TO_NO_XX_ID = getId(b'TO_NO_XX_TOKEN')
TO_NO_XX_TOKEN = getToken(b'TO_NO_XX_TOKEN').decode('UTF-8')
TO_PL_PL_ID = getId(b'TO_PL_PL_TOKEN')
TO_PL_PL_TOKEN = getToken(b'TO_PL_PL_TOKEN').decode('UTF-8')
TO_PT_XX_ID = getId(b'TO_PT_XX_TOKEN')
TO_PT_XX_TOKEN = getToken(b'TO_PT_XX_TOKEN').decode('UTF-8')
TO_RO_RO_ID = getId(b'TO_RO_RO_TOKEN')
TO_RO_RO_TOKEN = getToken(b'TO_RO_RO_TOKEN').decode('UTF-8')
TO_RU_RU_ID = getId(b'TO_RU_RU_TOKEN')
TO_RU_RU_TOKEN = getToken(b'TO_RU_RU_TOKEN').decode('UTF-8')
TO_SK_SK_ID = getId(b'TO_SK_SK_TOKEN')
TO_SK_SK_TOKEN = getToken(b'TO_SK_SK_TOKEN').decode('UTF-8')
TO_SL_SI_ID = getId(b'TO_SL_SI_TOKEN')
TO_SL_SI_TOKEN = getToken(b'TO_SL_SI_TOKEN').decode('UTF-8')
TO_SQ_AL_ID = getId(b'TO_SQ_AL_TOKEN')
TO_SQ_AL_TOKEN = getToken(b'TO_SQ_AL_TOKEN').decode('UTF-8')
TO_SV_SE_ID = getId(b'TO_SV_SE_TOKEN')
TO_SV_SE_TOKEN = getToken(b'TO_SV_SE_TOKEN').decode('UTF-8')
TO_SW_KE_ID = getId(b'TO_SW_KE_TOKEN')
TO_SW_KE_TOKEN = getToken(b'TO_SW_KE_TOKEN').decode('UTF-8')
TO_TA_IN_ID = getId(b'TO_TA_IN_TOKEN')
TO_TA_IN_TOKEN = getToken(b'TO_TA_IN_TOKEN').decode('UTF-8')
TO_TH_TH_ID = getId(b'TO_TH_TH_TOKEN')
TO_TH_TH_TOKEN = getToken(b'TO_TH_TH_TOKEN').decode('UTF-8')
TO_TL_XX_ID = getId(b'TO_TL_XX_TOKEN')
TO_TL_XX_TOKEN = getToken(b'TO_TL_XX_TOKEN').decode('UTF-8')
TO_TR_TR_ID = getId(b'TO_TR_TR_TOKEN')
TO_TR_TR_TOKEN = getToken(b'TO_TR_TR_TOKEN').decode('UTF-8')
TO_UK_UA_ID = getId(b'TO_UK_UA_TOKEN')
TO_UK_UA_TOKEN = getToken(b'TO_UK_UA_TOKEN').decode('UTF-8')
TO_VI_VN_ID = getId(b'TO_VI_VN_TOKEN')
TO_VI_VN_TOKEN = getToken(b'TO_VI_VN_TOKEN').decode('UTF-8')
TO_ZH_TW_ID = getId(b'TO_ZH_TW_TOKEN')
TO_ZH_TW_TOKEN = getToken(b'TO_ZH_TW_TOKEN').decode('UTF-8')
TO_ZH_CN_ID = getId(b'TO_ZH_CN_TOKEN')
TO_ZH_CN_TOKEN = getToken(b'TO_ZH_CN_TOKEN').decode('UTF-8')
