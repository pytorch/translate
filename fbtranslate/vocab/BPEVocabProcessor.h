#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "language_technology/neural_mt/fbtranslate/vocab/BaseVocabProcessor.h"

namespace facebook {
namespace language_technology {
namespace neural_mt {

class BPEVocabProcessor : public BaseVocabProcessor {
 public:
  explicit BPEVocabProcessor(std::vector<TokenAndCount>);

  int getId(const std::string& token) const override;
  std::pair<bool, TokenAndCount> getFullToken(int tokenId) const override;
  std::pair<std::vector<int>, TokenToIndexAlignment> numberize(
      const std::vector<std::string>& tokens) const override;

  std::pair<std::vector<std::string>, TokenToIndexAlignment> denumberize(
      const std::vector<int>& tokenIds) const override;

  size_t size() const override;

 private:
  std::vector<TokenAndCount> tokensAndCounts_;
  std::unordered_map<std::string, int> token2id_;
};

} // namespace neural_mt
} // namespace language_technology
} // namespace facebook
