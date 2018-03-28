#include "language_technology/neural_mt/fbtranslate/vocab/WordVocabProcessor.h"

#include <numeric>

#include <folly/Format.h>

#include "common/base/Exception.h"
#include "language_technology/neural_mt/fbtranslate/vocab/VocabConstants.h"

namespace facebook {
namespace language_technology {
namespace neural_mt {

WordVocabProcessor::WordVocabProcessor(
    std::vector<TokenAndCount> tokensAndCounts)
    : tokensAndCounts_(std::move(tokensAndCounts)) {
  if (VocabConstants::SPECIAL_TOKENS().size() >
      VocabConstants::MAX_SPECIAL_TOKENS) {
    FBEXCEPTION("There are too many special tokens.");
  }
  token2id_.reserve(
      VocabConstants::SPECIAL_TOKENS().size() + tokensAndCounts_.size());

  for (size_t i = 0; i < VocabConstants::SPECIAL_TOKENS().size(); ++i) {
    auto ret = token2id_.insert(
        std::make_pair(VocabConstants::SPECIAL_TOKENS()[i], i));
    if (ret.second == false) {
      FBEXCEPTION(
          folly::format(
              "Element {} already exists", VocabConstants::SPECIAL_TOKENS()[i])
              .str());
    }
  }
  for (size_t i = 0; i < tokensAndCounts_.size(); ++i) {
    auto ret = token2id_.insert(std::make_pair(
        tokensAndCounts_[i].first, i + VocabConstants::MAX_SPECIAL_TOKENS));
    if (ret.second == false) {
      FBEXCEPTION(
          folly::format("Element {} already exists", tokensAndCounts_[i].first)
              .str());
    }
  }
}

int WordVocabProcessor::getId(const std::string& token) const {
  auto found = token2id_.find(token);
  return (found != token2id_.end()) ? found->second
                                    : VocabConstants::INVALID_ID;
}

std::pair<bool, TokenAndCount> WordVocabProcessor::getFullToken(
    int tokenId) const {
  TokenAndCount undefinedToken;
  undefinedToken.first = VocabConstants::getToken("UNDEFINED_TOKEN");
  undefinedToken.second = 0;
  if (tokenId < 0) {
    return std::make_pair(false, undefinedToken);
  } else if (tokenId < VocabConstants::MAX_SPECIAL_TOKENS) {
    if (tokenId >= VocabConstants::SPECIAL_TOKENS().size()) {
      return std::make_pair(true, undefinedToken);
    } else {
      TokenAndCount word;
      word.first = VocabConstants::SPECIAL_TOKENS()[tokenId];
      word.second = 0;
      return std::make_pair(true, word);
    }
  } else if (
      tokenId < VocabConstants::MAX_SPECIAL_TOKENS + tokensAndCounts_.size()) {
    return std::make_pair(
        true, tokensAndCounts_[tokenId - VocabConstants::MAX_SPECIAL_TOKENS]);
  } else {
    return std::make_pair(false, undefinedToken);
  }
}

std::pair<std::vector<int>, TokenToIndexAlignment>
WordVocabProcessor::numberize(const std::vector<std::string>& tokens) const {
  std::vector<int> res;
  for (const auto& token : tokens) {
    auto found = token2id_.find(token);
    if (found != token2id_.end()) {
      res.emplace_back(found->second);
    } else {
      res.emplace_back(VocabConstants::getId("UNK_TOKEN"));
    }
  }
  TokenToIndexAlignment alignment(tokens.size());
  std::iota(alignment.begin(), alignment.end(), 0);
  return std::make_pair(std::move(res), std::move(alignment));
}

std::pair<std::vector<std::string>, TokenToIndexAlignment>
WordVocabProcessor::denumberize(const std::vector<int>& tokenIds) const {
  std::vector<std::string> res;
  for (const auto& tokenId : tokenIds) {
    auto token = getToken(tokenId);
    if (!token.first) {
      FBEXCEPTION(folly::format("Out of range id: {}", tokenId).str());
    }
    res.emplace_back(token.second);
  }
  TokenToIndexAlignment alignment(tokenIds.size());
  std::iota(alignment.begin(), alignment.end(), 0);
  return std::make_pair(std::move(res), std::move(alignment));
}

size_t WordVocabProcessor::size() const {
  return VocabConstants::MAX_SPECIAL_TOKENS + tokensAndCounts_.size();
}
} // namespace neural_mt
} // namespace language_technology
} // namespace facebook
