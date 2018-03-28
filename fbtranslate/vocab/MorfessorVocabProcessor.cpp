#include "language_technology/neural_mt/fbtranslate/vocab/MorfessorVocabProcessor.h"

#include <folly/Format.h>

#include "common/base/Exception.h"
#include "common/strings/icu/ICUString.h"
#include "language_technology/neural_mt/fbtranslate/vocab/VocabConstants.h"
#include "language_technology/yoda/YodaTagUtil.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace facebook {
namespace language_technology {
namespace neural_mt {

MorfessorVocabProcessor::MorfessorVocabProcessor(MorfessorVocab vocab)
    : morfessorVocab_(std::move(vocab)) {
  auto yoda_tags_set = yoda::getYodaTags();
  std::vector<std::string> yoda_tags(
      yoda_tags_set.begin(), yoda_tags_set.end());

  if (VocabConstants::SPECIAL_TOKENS().size() + yoda_tags.size() >
      VocabConstants::MAX_SPECIAL_TOKENS) {
    FBEXCEPTION("There are too many special tokens.");
  }
  morf2id_.reserve(
      VocabConstants::SPECIAL_TOKENS().size() + yoda_tags.size() +
      morfessorVocab_.morfs.size());

  Word unk_word;
  unk_word.token = VocabConstants::getToken("UNDEFINED_TOKEN");
  unk_word.count = 0;
  id2token_.resize(
      VocabConstants::MAX_SPECIAL_TOKENS + morfessorVocab_.morfs.size(),
      unk_word);

  for (size_t i = 0; i < VocabConstants::SPECIAL_TOKENS().size(); ++i) {
    auto ret =
        morf2id_.insert(std::make_pair(VocabConstants::SPECIAL_TOKENS()[i], i));
    if (ret.second == false) {
      FBEXCEPTION(folly::format(
                      "Special token {} already exists",
                      VocabConstants::SPECIAL_TOKENS()[i])
                      .str());
    }

    Word word;
    word.token = VocabConstants::SPECIAL_TOKENS()[i];
    word.count = 0;
    id2token_[i] = word;
  }

  for (size_t i = 0; i < yoda_tags.size(); ++i) {
    auto ret = morf2id_.insert(std::make_pair(
        yoda_tags[i], i + VocabConstants::SPECIAL_TOKENS().size()));
    if (ret.second == false) {
      FBEXCEPTION(
          folly::format("Element {} already exists", yoda_tags[i]).str());
    }

    Word word;
    word.token = yoda_tags[i];
    word.count = 0;
    id2token_[i + VocabConstants::SPECIAL_TOKENS().size()] = word;
  }

  for (size_t i = 0; i < morfessorVocab_.morfs.size(); ++i) {
    // Because the yoda_tags may be in the words already, continue if this word
    // is a yoda_tag.
    if (yoda_tags_set.find(morfessorVocab_.morfs[i].token) !=
        yoda_tags_set.end()) {
      continue;
    }
    auto ret = morf2id_.insert(std::make_pair(
        morfessorVocab_.morfs[i].token,
        i + VocabConstants::MAX_SPECIAL_TOKENS));
    if (ret.second == false) {
      FBEXCEPTION(folly::format(
                      "Non-special element {} already exists",
                      morfessorVocab_.morfs[i].token)
                      .str());
    }

    id2token_[i + VocabConstants::MAX_SPECIAL_TOKENS] =
        morfessorVocab_.morfs[i];
  }
}

int MorfessorVocabProcessor::getId(const std::string& token) const {
  auto found = morf2id_.find(token);
  return (found != morf2id_.end()) ? found->second : VocabConstants::INVALID_ID;
}

std::pair<bool, Word> MorfessorVocabProcessor::getFullToken(int tokenId) const {
  if (tokenId < 0 || tokenId >= id2token_.size()) {
    return std::make_pair(false, Word());
  }

  return std::make_pair(true, id2token_[tokenId]);
}

std::pair<std::vector<int>, TokenToIndexAlignment>
MorfessorVocabProcessor::numberize(
    const std::vector<std::string>& tokens) const {
  std::vector<int> res;
  TokenToIndexAlignment alignment;

  auto log_tokens = log(morfessorVocab_.token_count);

  for (const auto& token : tokens) {
    // If the entire token exists as a special token, add it directly
    int token_id = getId(token);
    if (token_id != VocabConstants::INVALID_ID &&
        token_id < VocabConstants::MAX_SPECIAL_TOKENS) {
      alignment.emplace_back(res.size());
      res.emplace_back(VocabConstants::getId("START_WORD_TOKEN"));
      res.emplace_back(token_id);
      continue;
    }

    icu::UnicodeString utoken = icu::UnicodeString::fromUTF8(token);

    // cost[i] represents the lowest cost of the morfs from [0, i)
    std::vector<int> cost(utoken.length() + 1, -1);
    // parent[i] represents the starting position of the last morf in the
    // sequence that gives the lowest cost
    std::vector<int> parent(utoken.length() + 1, -1);
    // pos_token[i] gives the last morf in the sequence which gives the lowest
    // cost
    std::vector<std::string> pos_token(utoken.length() + 1, "");

    cost[0] = 0;
    parent[0] = 0;

    // Iterate over the possible end positions (non-inclusive) and identify the
    // relevant information for each end position.
    for (int i = 1; i < utoken.length() + 1; ++i) {
      // Consider every possible starting position for a given ending position.
      for (int j = 0; j < i; ++j) {
        // Continue if it is impossible to get to the the start position.
        if (parent[j] == -1) {
          continue;
        }

        icu::UnicodeString subToken(utoken, j, i - j);

        // The cost we are trying to minimize is the sum of the negative
        // log likelihood of the morpheme probabilities. The probability
        // of any given morpheme is [frequency of that morpheme] divided by
        // [total frequency of all morphemes] (this is what log_tokens)
        // represents.
        std::string subTokenUtf8;
        subToken.toUTF8String(subTokenUtf8);
        int cur_id = getId(subTokenUtf8);
        int cur_cost = cost[j];
        std::string cur_token;
        if (cur_id != VocabConstants::INVALID_ID) {
          auto word = getFullToken(cur_id);
          // Add the negative log likelihood of this morpheme.
          cur_cost += log_tokens - log(word.second.count);
          cur_token = word.second.token;
        } else {
          // If it is impossible to find the morpheme, assuming that
          // the probability of it is 1/N.
          cur_cost += log_tokens * (i - j);
          cur_token = VocabConstants::getToken("UNK_TOKEN");
        }

        if (cost[i] == -1 || cur_cost < cost[i]) {
          cost[i] = cur_cost;
          parent[i] = j;
          pos_token[i] = cur_token;
        }
      }
    }

    // Iterate backwards over the tokens and repeatedly go to the parent that
    // gives the lowest cost. This way we can use our dynamic programming
    // vector to reconstruct the optimal (most likely)  sequence of morphemes.
    int cur = utoken.length();
    std::vector<std::string> encoded_token;
    while (cur != 0) {
      encoded_token.emplace_back(pos_token[cur]);
      cur = parent[cur];
    }

    encoded_token.emplace_back(VocabConstants::getToken("START_WORD_TOKEN"));
    std::reverse(encoded_token.begin(), encoded_token.end());

    // Add all the tokens that encode this word to the result. If we have
    // consecutive UNK tokens -- only add one.
    alignment.emplace_back(res.size());
    std::string prev = "";
    for (const auto& etoken : encoded_token) {
      if (prev == VocabConstants::getToken("UNK_TOKEN") &&
          etoken == VocabConstants::getToken("UNK_TOKEN")) {
        continue;
      }

      res.emplace_back(getId(etoken));
      prev = etoken;
    }
  }
  return std::make_pair(std::move(res), std::move(alignment));
}

size_t MorfessorVocabProcessor::size() const {
  return VocabConstants::MAX_SPECIAL_TOKENS + morfessorVocab_.morfs.size();
}
} // namespace neural_mt
} // namespace language_technology
} // namespace facebook
