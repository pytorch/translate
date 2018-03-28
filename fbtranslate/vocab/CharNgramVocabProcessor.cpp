#include "language_technology/neural_mt/fbtranslate/vocab/CharNgramVocabProcessor.h"

#include <folly/Format.h>

#include "common/base/Exception.h"
#include "common/strings/icu/ICUString.h"
#include "language_technology/neural_mt/fbtranslate/vocab/VocabConstants.h"
#include "language_technology/yoda/YodaTagUtil.h"

#include <vector>

namespace facebook {
namespace language_technology {
namespace neural_mt {

CharNgramVocabProcessor::CharNgramVocabProcessor(CharNgramVocab charNgramVocab)
    : charNgramVocab_(std::move(charNgramVocab)) {
  auto yoda_tags_set = yoda::getYodaTags();
  std::vector<std::string> yoda_tags(
      yoda_tags_set.begin(), yoda_tags_set.end());

  if (VocabConstants::SPECIAL_TOKENS().size() + yoda_tags.size() >
      VocabConstants::MAX_SPECIAL_TOKENS) {
    FBEXCEPTION("There are too many special tokens.");
  }

  Word unk_word;
  unk_word.token = VocabConstants::getToken("UNDEFINED_TOKEN");
  unk_word.count = 0;
  id2token_.resize(
      VocabConstants::MAX_SPECIAL_TOKENS + charNgramVocab_.words.size() +
          charNgramVocab_.ngrams.size(),
      unk_word);

  word2id_.reserve(
      VocabConstants::SPECIAL_TOKENS().size() + yoda_tags.size() +
      charNgramVocab_.words.size());

  for (size_t i = 0; i < VocabConstants::SPECIAL_TOKENS().size(); ++i) {
    auto ret =
        word2id_.insert(std::make_pair(VocabConstants::SPECIAL_TOKENS()[i], i));
    if (ret.second == false) {
      FBEXCEPTION(
          folly::format(
              "Element {} already exists", VocabConstants::SPECIAL_TOKENS()[i])
              .str());
    }

    Word word;
    word.token = VocabConstants::SPECIAL_TOKENS()[i];
    word.count = 0;
    id2token_[i] = word;
  }

  for (size_t i = 0; i < yoda_tags.size(); ++i) {
    auto ret = word2id_.insert(std::make_pair(
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

  for (size_t i = 0; i < charNgramVocab_.words.size(); ++i) {
    // Because the yoda_tags may be in the words already, continue if this word
    // is a yoda_tag.
    if (yoda_tags_set.find(charNgramVocab_.words[i].token) !=
        yoda_tags_set.end()) {
      continue;
    }

    auto ret = word2id_.insert(std::make_pair(
        charNgramVocab_.words[i].token,
        i + VocabConstants::MAX_SPECIAL_TOKENS));
    if (ret.second == false) {
      FBEXCEPTION(
          folly::format(
              "Element {} already exists", charNgramVocab_.words[i].token)
              .str());
    }

    id2token_[i + VocabConstants::MAX_SPECIAL_TOKENS] =
        charNgramVocab_.words[i];
  }
  ngram2id_.reserve(charNgramVocab_.ngrams.size());
  for (size_t i = 0; i < charNgramVocab_.ngrams.size(); ++i) {
    auto ret = ngram2id_.insert(std::make_pair(
        charNgramVocab_.ngrams[i].token,
        i + VocabConstants::MAX_SPECIAL_TOKENS + charNgramVocab_.words.size()));
    if (ret.second == false) {
      FBEXCEPTION(
          folly::format(
              "Element {} already exists", charNgramVocab_.ngrams[i].token)
              .str());
    }

    int ind =
        i + VocabConstants::MAX_SPECIAL_TOKENS + charNgramVocab_.words.size();
    id2token_[ind] = charNgramVocab_.ngrams[i];
  }
}

int CharNgramVocabProcessor::getId(const std::string& token) const {
  std::string new_token = token;
  bool has_control_char = false;
  int control_length =
      std::string(VocabConstants::getToken("CONTROL_TOKEN")).length();
  if (token.length() > 0 &&
      token.substr(0, control_length) ==
          VocabConstants::getToken("CONTROL_TOKEN")) {
    // Remove the control token
    new_token.erase(0, control_length);
    has_control_char = true;
  }

  auto found = word2id_.find(new_token);
  if (found == word2id_.end() || has_control_char) {
    found = ngram2id_.find(new_token);
    return (found != ngram2id_.end()) ? found->second
                                      : VocabConstants::INVALID_ID;
  } else {
    return found->second;
  }
}

std::pair<bool, Word> CharNgramVocabProcessor::getTokenWithControl(
    int tokenId) const {
  if (tokenId < 0 || tokenId >= id2token_.size()) {
    return std::make_pair(false, Word());
  }

  Word word = id2token_[tokenId];
  if (tokenId >=
      VocabConstants::MAX_SPECIAL_TOKENS + charNgramVocab_.words.size()) {
    Word new_word;
    new_word.token = VocabConstants::getToken("CONTROL_TOKEN") + word.token;
    new_word.count = word.count;
    word = new_word;
  }

  return std::make_pair(true, word);
}

std::pair<bool, Word> CharNgramVocabProcessor::getFullToken(int tokenId) const {
  if (tokenId < 0 || tokenId >= id2token_.size()) {
    return std::make_pair(false, Word());
  }

  return std::make_pair(true, id2token_[tokenId]);
}

std::pair<std::vector<int>, TokenToIndexAlignment>
CharNgramVocabProcessor::numberize(
    const std::vector<std::string>& tokens) const {
  std::vector<int> result;
  TokenToIndexAlignment alignment;
  int position = -1;
  for (const auto& token : tokens) {
    auto found = word2id_.find(token);
    if (found != word2id_.end()) {
      result.emplace_back(found->second);
      alignment.emplace_back(++position);
    } else {
      icu::UnicodeString ngram("");
      result.emplace_back(VocabConstants::getId("START_WORD_TOKEN"));
      alignment.emplace_back(++position);
      icu::UnicodeString uToken = icu::UnicodeString::fromUTF8(token);
      for (int i = 0; i < uToken.length(); ++i) {
        ngram.append(uToken[i]);
        if ((i + 1) % charNgramVocab_.ngram_size == 0) {
          std::string ngramUtf8;
          ngram.toUTF8String(ngramUtf8);
          auto ngramFound = ngram2id_.find(ngramUtf8);
          if (ngramFound != ngram2id_.end()) {
            result.emplace_back(ngramFound->second);
          } else {
            result.emplace_back(VocabConstants::getId("UNK_TOKEN"));
          }
          ngram = "";
          ++position;
        }
      }
      if (!ngram.isEmpty()) {
        auto ngramUtf8 = facebook::strings::icuStringToUTF8(ngram);
        auto ngramFound = ngram2id_.find(ngramUtf8);
        if (ngramFound != ngram2id_.end()) {
          result.emplace_back(ngramFound->second);
        } else {
          result.emplace_back(VocabConstants::getId("UNK_TOKEN"));
        }
        ++position;
      }
    }
  }
  return std::make_pair(std::move(result), std::move(alignment));
}

std::pair<std::vector<std::string>, TokenToIndexAlignment>
CharNgramVocabProcessor::denumberize(const std::vector<int>& tokenIds) const {
  std::vector<std::string> tokens;
  std::string token;
  TokenToIndexAlignment alignment;
  int position = -1;
  bool seenStartToken = false;

  auto yoda_tags_set = yoda::getYodaTags();
  std::vector<std::string> yoda_tags(
      yoda_tags_set.begin(), yoda_tags_set.end());

  for (const auto& tokenId : tokenIds) {
    if (tokenId < 0 || tokenId >= id2token_.size()) {
      FBEXCEPTION(folly::format("Out of range id: {}", tokenId).str());
    }

    bool is_ngram = tokenId >=
        VocabConstants::MAX_SPECIAL_TOKENS + charNgramVocab_.words.size();
    if (!is_ngram && !token.empty()) {
      tokens.emplace_back(token);
      token.clear();
      // We set seenStartToken to false every time we consume a token
      seenStartToken = false;
    }

    ++position;
    if (tokenId == VocabConstants::getId("START_WORD_TOKEN")) {
      seenStartToken = true;
      alignment.emplace_back(position);
    } else if (!is_ngram) {
      tokens.emplace_back(id2token_[tokenId].token);
      alignment.emplace_back(position);
      // We set seenStartToken to false every time we consume a token
      seenStartToken = false;
    } else {
      if (seenStartToken == false) {
        // We see an ngram token without seeing SOW ID. Let's assume we saw SOW
        // and add position to alignment
        alignment.emplace_back(position);
        seenStartToken = true;
      }
      token.append(id2token_[tokenId].token);
    }
  }

  if (!token.empty()) {
    LOG(INFO) << "Adding token in the end";
    tokens.emplace_back(token);
  }
  return std::make_pair(std::move(tokens), std::move(alignment));
}

size_t CharNgramVocabProcessor::size() const {
  return VocabConstants::MAX_SPECIAL_TOKENS + charNgramVocab_.words.size() +
      charNgramVocab_.ngrams.size();
}
} // namespace neural_mt
} // namespace language_technology
} // namespace facebook
