#include "language_technology/neural_mt/fbtranslate/vocab/VocabConstants.h"

#include <stdint.h>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>
namespace facebook {
namespace language_technology {
namespace neural_mt {

/*
  Add new special tokens here. If you want to add a token that has a token value
  but you don't want to add it to the SPECIAL_TOKENS() list or the
  SPECIAL_TOKENS_TO_ID map, specify INVALID_ID instead of an actual ID value
  (See CONTROL_TOKEN for an example).

  Make sure to also expose your token in the Python wrapper vocab_constants.pyx
*/
const std::vector<std::pair<std::string, std::pair<const char*, std::int32_t>>>&
VocabConstants::SPECIAL_TOKENS_MAP() {
  static const std::vector<
      std::pair<std::string, std::pair<const char*, std::int32_t>>>
      SPECIAL_TOKENS_MAP_ = {
          {"PAD_TOKEN", {"_PAD", 0}},
          {"GO_TOKEN", {"_GO", 1}},
          {"EOS_TOKEN", {"_EOS", 2}},
          {"UNK_TOKEN", {"_UNK", 3}},
          {"START_WORD_TOKEN", {"_SOW", 4}},
          {"END_WORD_TOKEN", {"_EOW", 5}},
          {"COPY_TOKEN", {"_COPY", 6}},
          {"UNDEFINED_TOKEN", {"_UNDEFINED", 7}},
          {"CONTROL_TOKEN", {"_CTRL", VocabConstants::INVALID_ID}},
          {"TO_AF_ZA_TOKEN", {"_TO_af_ZA", 8}},
          {"TO_AR_AR_TOKEN", {"_TO_ar_AR", 9}},
          {"TO_BG_BG_TOKEN", {"_TO_bg_BG", 10}},
          {"TO_BN_IN_TOKEN", {"_TO_bn_IN", 11}},
          {"TO_BR_FR_TOKEN", {"_TO_br_FR", 12}},
          {"TO_BS_BA_TOKEN", {"_TO_bs_BA", 13}},
          {"TO_CA_ES_TOKEN", {"_TO_ca_ES", 14}},
          {"TO_CS_CZ_TOKEN", {"_TO_cs_CZ", 15}},
          {"TO_CY_GB_TOKEN", {"_TO_cy_GB", 16}},
          {"TO_DA_DK_TOKEN", {"_TO_da_DK", 17}},
          {"TO_DE_DE_TOKEN", {"_TO_de_DE", 18}},
          {"TO_EL_GR_TOKEN", {"_TO_el_GR", 19}},
          {"TO_EN_XX_TOKEN", {"_TO_en_XX", 20}},
          {"TO_ES_XX_TOKEN", {"_TO_es_XX", 21}},
          {"TO_ET_EE_TOKEN", {"_TO_et_EE", 22}},
          {"TO_EU_ES_TOKEN", {"_TO_eu_ES", 23}},
          {"TO_FA_IR_TOKEN", {"_TO_fa_IR", 24}},
          {"TO_FI_FI_TOKEN", {"_TO_fi_FI", 25}},
          {"TO_FR_XX_TOKEN", {"_TO_fr_XX", 26}},
          {"TO_HE_IL_TOKEN", {"_TO_he_IL", 27}},
          {"TO_HI_IN_TOKEN", {"_TO_hi_IN", 28}},
          {"TO_HR_HR_TOKEN", {"_TO_hr_HR", 29}},
          {"TO_HU_HU_TOKEN", {"_TO_hu_HU", 30}},
          {"TO_ID_ID_TOKEN", {"_TO_id_ID", 31}},
          {"TO_IT_IT_TOKEN", {"_TO_it_IT", 32}},
          {"TO_JA_XX_TOKEN", {"_TO_ja_XX", 33}},
          {"TO_KO_KR_TOKEN", {"_TO_ko_KR", 34}},
          {"TO_LT_LT_TOKEN", {"_TO_lt_LT", 35}},
          {"TO_LV_LV_TOKEN", {"_TO_lv_LV", 36}},
          {"TO_MK_MK_TOKEN", {"_TO_mk_MK", 37}},
          {"TO_MS_MY_TOKEN", {"_TO_ms_MY", 38}},
          {"TO_NL_XX_TOKEN", {"_TO_nl_XX", 39}},
          {"TO_NO_XX_TOKEN", {"_TO_no_XX", 40}},
          {"TO_PL_PL_TOKEN", {"_TO_pl_PL", 41}},
          {"TO_PT_XX_TOKEN", {"_TO_pt_XX", 42}},
          {"TO_RO_RO_TOKEN", {"_TO_ro_RO", 43}},
          {"TO_RU_RU_TOKEN", {"_TO_ru_RU", 44}},
          {"TO_SK_SK_TOKEN", {"_TO_sk_SK", 45}},
          {"TO_SL_SI_TOKEN", {"_TO_sl_SI", 46}},
          {"TO_SQ_AL_TOKEN", {"_TO_sq_AL", 47}},
          {"TO_SV_SE_TOKEN", {"_TO_sv_SE", 48}},
          {"TO_SW_KE_TOKEN", {"_TO_sw_KE", 49}},
          {"TO_TA_IN_TOKEN", {"_TO_ta_IN", 50}},
          {"TO_TH_TH_TOKEN", {"_TO_th_TH", 51}},
          {"TO_TL_XX_TOKEN", {"_TO_tl_XX", 52}},
          {"TO_TR_TR_TOKEN", {"_TO_tr_TR", 53}},
          {"TO_UK_UA_TOKEN", {"_TO_uk_UA", 54}},
          {"TO_VI_VN_TOKEN", {"_TO_vi_VN", 55}},
          {"TO_ZH_TW_TOKEN", {"_TO_zh_TW", 56}},
          {"TO_ZH_CN_TOKEN", {"_TO_zh_CN", 57}}};
  return SPECIAL_TOKENS_MAP_;
}

// declare int/ string constants to prevent missing symbols in Cython P59254227
constexpr int32_t const VocabConstants::MAX_SPECIAL_TOKENS;
constexpr int32_t const VocabConstants::INVALID_ID;

constexpr char const* VocabConstants::BPE_VOCAB_TYPE;
constexpr char const* VocabConstants::CHAR_NGRAM_VOCAB_TYPE;
constexpr char const* VocabConstants::MORFESSOR_VOCAB_TYPE;
constexpr char const* VocabConstants::WORDPIECE_VOCAB_TYPE;
constexpr char const* VocabConstants::WORD_VOCAB_TYPE;

// public SPECIAL_TOKENS() and SPECIAL_TOKENS_TO_ID() data structures
// Note: "static local objects are constructed the first time control flows over
// their declaration (only)" https://isocpp.org/wiki/faq/ctors#static-init-order
const std::vector<std::string> VocabConstants::SPECIAL_TOKENS() {
  static const std::vector<std::string> SPECIAL_TOKENS_ =
      *VocabConstants::specialTokensInitializer();
  return SPECIAL_TOKENS_;
}
const std::unordered_map<std::string, int32_t>
VocabConstants::SPECIAL_TOKENS_TO_ID() {
  static const std::unordered_map<std::string, int32_t> SPECIAL_TOKENS_TO_ID_ =
      *VocabConstants::specialTokensToIdInitializer();
  return SPECIAL_TOKENS_TO_ID_;
}

std::vector<std::string>* VocabConstants::specialTokensInitializer() {
  std::vector<std::string>* v = new std::vector<std::string>();
  for (const auto& tokenInfo : VocabConstants::SPECIAL_TOKENS_MAP()) {
    std::pair<const char*, int32_t> tokenValuePair = tokenInfo.second;
    if (tokenValuePair.second != -1) {
      v->push_back(tokenValuePair.first);
    }
  }
  return v;
}

std::unordered_map<std::string, int32_t>*
VocabConstants::specialTokensToIdInitializer() {
  std::unordered_map<std::string, int32_t>* m =
      new std::unordered_map<std::string, int32_t>();
  for (const auto& tokenInfo : VocabConstants::SPECIAL_TOKENS_MAP()) {
    std::pair<const char*, int32_t> tokenValuePair = tokenInfo.second;
    if (tokenValuePair.second != -1) {
      m->insert(tokenValuePair);
    }
  }
  return m;
}

// public methods to return token value and token id given a token name
const char* VocabConstants::getToken(std::string tokenName) {
  auto it = std::find_if(
      VocabConstants::SPECIAL_TOKENS_MAP().begin(),
      VocabConstants::SPECIAL_TOKENS_MAP().end(),
      [tokenName](auto element) { return element.first == tokenName; });
  return it->second.first;
}

int32_t VocabConstants::getId(std::string tokenName) {
  auto it = std::find_if(
      VocabConstants::SPECIAL_TOKENS_MAP().begin(),
      VocabConstants::SPECIAL_TOKENS_MAP().end(),
      [tokenName](auto element) { return element.first == tokenName; });
  return it->second.second;
}

} // namespace neural_mt
} // namespace language_technology
} // namespace facebook
