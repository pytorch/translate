#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "language_technology/neural_mt/fbtranslate/vocab/BaseVocabProcessor.h"
#include "language_technology/neural_mt/vocab/if/gen-cpp2/vocab_types.h"

namespace facebook {
namespace language_technology {
namespace neural_mt {

class WordpieceVocabProcessor : public BaseVocabProcessor {
 public:
  explicit WordpieceVocabProcessor(WordpieceVocab vocab);

  int getId(const std::string& token) const override;
  std::pair<bool, Word> getFullToken(int tokenId) const override;
  std::pair<std::vector<int>, TokenToIndexAlignment> numberize(
      const std::vector<std::string>& tokens) const override;

  size_t size() const override;

 private:
  WordpieceVocab wordpieceVocab_;
  std::unordered_map<std::string, int> token2id_;
};

} // namespaces
}
}
