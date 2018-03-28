#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace facebook {
namespace language_technology {
namespace neural_mt {
class VocabConstants {
 private:
  static const std::vector<
      std::pair<std::string, std::pair<const char*, std::int32_t>>>&
  SPECIAL_TOKENS_MAP();
  static std::vector<std::string>* specialTokensInitializer();
  static std::unordered_map<std::string, int32_t>*
  specialTokensToIdInitializer();

 public:
  static const std::vector<std::string> SPECIAL_TOKENS();
  static const std::unordered_map<std::string, int32_t> SPECIAL_TOKENS_TO_ID();
  static const char* getToken(std::string tokenName);
  static int32_t getId(std::string tokenName);

  // If you add an integer / string constant below, you must declare them in
  // VocabConstants.cpp and expose them in vocab_constants.pyx
  static constexpr int32_t const INVALID_ID = -1;
  static constexpr int32_t const MAX_SPECIAL_TOKENS = 100;

  static constexpr char const* BPE_VOCAB_TYPE = "bpe_vocab";
  static constexpr char const* CHAR_NGRAM_VOCAB_TYPE = "char_ngram_vocab";
  static constexpr char const* MORFESSOR_VOCAB_TYPE = "morfessor_vocab";
  static constexpr char const* WORDPIECE_VOCAB_TYPE = "wordpiece_vocab";
  static constexpr char const* WORD_VOCAB_TYPE = "word_vocab";
};
} // namespace neural_mt
} // namespace language_technology
} // namespace facebook
