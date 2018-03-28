#include "language_technology/neural_mt/fbtranslate/vocab/BPEVocabProcessor.h"
#include <folly/Format.h>

#include "common/base/Exception.h"
#include "common/strings/icu/ICUString.h"
#include "language_technology/neural_mt/fbtranslate/vocab/VocabConstants.h"
#include "language_technology/yoda/YodaTagUtil.h"

#include <vector>

namespace facebook {
namespace language_technology {
namespace neural_mt {

BPEVocabProcessor::BPEVocabProcessor(std::vector<TokenAndCount> tokensAndCounts)
    : tokensAndCounts_(std::move(tokensAndCounts)) {
  auto yoda_tags_set = yoda::getYodaTags();
  std::vector<std::string> yoda_tags(
      yoda_tags_set.begin(), yoda_tags_set.end());

  if (VocabConstants::SPECIAL_TOKENS().size() + yoda_tags.size() >
      VocabConstants::MAX_SPECIAL_TOKENS) {
    FBEXCEPTION("There are too many special tokens.");
  }
  token2id_.reserve(
      VocabConstants::SPECIAL_TOKENS().size() + yoda_tags.size() +
      tokensAndCounts_.size());

  TokenAndCount unk_word =
      TokenAndCount(VocabConstants::getToken("UNDEFINED_TOKEN"), 0);
  id2token_.resize(
      VocabConstants::MAX_SPECIAL_TOKENS + tokensAndCounts_.size(), unk_word);

  for (size_t i = 0; i < VocabConstants::SPECIAL_TOKENS().size(); ++i) {
    auto ret = token2id_.insert(
        std::make_pair(VocabConstants::SPECIAL_TOKENS()[i], i));
    if (ret.second == false) {
      FBEXCEPTION(folly::format(
                      "Special token {} already exists",
                      VocabConstants::SPECIAL_TOKENS()[i])
                      .str());
    }

    TokenAndCount word;
    word.first = VocabConstants::SPECIAL_TOKENS()[i];
    word.second = 0;
    id2token_[i] = word;
  }

  for (size_t i = 0; i < yoda_tags.size(); ++i) {
    auto ret = token2id_.insert(std::make_pair(
        yoda_tags[i], i + VocabConstants::SPECIAL_TOKENS().size()));
    if (ret.second == false) {
      FBEXCEPTION(
          folly::format("Element {} already exists", yoda_tags[i]).str());
    }

    TokenAndCount word;
    word.first = yoda_tags[i];
    word.second = 0;
    id2token_[i + VocabConstants::SPECIAL_TOKENS().size()] = word;
  }

  for (size_t i = 0; i < tokensAndCounts_.size(); ++i) {
    // Because the yoda_tags may be in the words already, continue if this word
    // is a yoda_tag.
    if (yoda_tags_set.find(tokensAndCounts_[i].first) != yoda_tags_set.end()) {
      continue;
    }

    auto ret = token2id_.insert(std::make_pair(
        tokensAndCounts_[i].first, i + VocabConstants::MAX_SPECIAL_TOKENS));
    if (ret.second == false) {
      FBEXCEPTION(folly::format(
                      "Non-special element {} already exists",
                      tokensAndCounts_[i].first)
                      .str());
    }

    id2token_[i + VocabConstants::MAX_SPECIAL_TOKENS] = tokensAndCounts_[i];
  }
}

int BPEVocabProcessor::getId(const std::string& token) const {
  auto found = token2id_.find(token);
  return (found != token2id_.end()) ? found->second
                                    : VocabConstants::INVALID_ID;
}

std::pair<bool, TokenAndCount> BPEVocabProcessor::getFullToken(
    int tokenId) const {
  if (tokenId < 0 || tokenId >= id2token_.size()) {
    return std::make_pair(
        false, TokenAndCount(VocabConstants::getToken("UNDEFINED_TOKEN"), 0));
  }

  return std::make_pair(true, id2token_[tokenId]);
}

bool endsWith(const std::string& word, const char* ending) {
  std::string endingStr(ending);
  if (word.length() >= endingStr.length()) {
    return (
        0 ==
        word.compare(
            word.length() - endingStr.length(), endingStr.length(), ending));
  } else {
    return false;
  }
}

std::pair<std::vector<int>, TokenToIndexAlignment> BPEVocabProcessor::numberize(
    const std::vector<std::string>& tokens) const {
  std::vector<int> res;
  TokenToIndexAlignment alignment;
  for (const auto& token : tokens) {
    // If the entire token exists in the corpus, skip BPE and add it directly.
    // Entire tokens always exist in the vocablary in the form `word + _EOW`
    if (getId(token + VocabConstants::getToken("END_WORD_TOKEN")) !=
        VocabConstants::INVALID_ID) {
      alignment.emplace_back(res.size());
      res.emplace_back(
          getId(token + VocabConstants::getToken("END_WORD_TOKEN")));
      continue;
    }

    std::vector<std::string> encoded_token;

    icu::UnicodeString utoken = icu::UnicodeString::fromUTF8(token);

    // For the word we're currently encoding: create a vector of character
    // tokens. If a character does not exist as a token -- it does not exist
    // as a subset of any token -- so we skip the novel character
    for (int i = 0; i < utoken.length(); ++i) {
      icu::UnicodeString ucur_character(utoken[i]);
      std::string cur_character;
      ucur_character.toUTF8String(cur_character);
      if (i == utoken.length() - 1) {
        cur_character =
            cur_character + VocabConstants::getToken("END_WORD_TOKEN");
      }
      if (getId(cur_character) != VocabConstants::INVALID_ID) {
        encoded_token.emplace_back(cur_character);
      } else {
        LOG(INFO) << "Skipping novel character not in vocabulary: "
                  << cur_character;
      }
    }

    if (encoded_token.size() > 0) {
      // This handles a rare edge case
      // If the last character in utoken is novel, we skip it and hence the last
      // character of utoken would not have an _EOW suffix
      // This is a violation of our assumption that all words end with
      // characters with fused _EOW, so we add it back
      if (!endsWith(
              encoded_token[encoded_token.size() - 1],
              VocabConstants::getToken("END_WORD_TOKEN"))) {
        encoded_token[encoded_token.size() - 1] +=
            VocabConstants::getToken("END_WORD_TOKEN");
      }
    } else {
      // Skip token if it has only novel characters
      // the deleted token still gets an alignment to the position of the
      // last id corresponding to the previous token
      if (res.empty()) { // token deleted corresponds to first token
        alignment.emplace_back(0);
      } else {
        alignment.emplace_back(res.size() - 1);
      }
      continue;
    }

    while (true) {
      int cur_id;
      int best_id = size() + 1;
      int best_ind = -1;

      // Find the most frequent bigram of tokens. Tokens with lower IDs are
      // more (or equally) frequent than tokens with higher IDs.
      for (int i = 0; i < encoded_token.size() - 1; ++i) {
        cur_id = getId(encoded_token[i] + encoded_token[i + 1]);
        if (cur_id != VocabConstants::INVALID_ID && cur_id < best_id) {
          best_id = cur_id;
          best_ind = i;
        }
      }

      if (best_ind == -1) {
        break;
      }

      // Replace the first occurence of the best token in the string.
      std::string best_token =
          encoded_token[best_ind] + encoded_token[best_ind + 1];
      encoded_token.erase(encoded_token.begin() + best_ind);
      encoded_token[best_ind] = best_token;

      if (encoded_token.size() == 1) {
        break;
      }
    }

    // Add all the tokens that encode this word to the result
    alignment.emplace_back(res.size());
    for (const auto& etoken : encoded_token) {
      res.emplace_back(getId(etoken));
    }
  }
  return std::make_pair(std::move(res), std::move(alignment));
}

std::pair<std::vector<std::string>, TokenToIndexAlignment>
BPEVocabProcessor::denumberize(const std::vector<int>& tokenIds) const {
  std::vector<std::string> tokens;
  std::string token;
  TokenToIndexAlignment alignment;
  int position = 0;

  // Adding alignment position for the first word
  alignment.emplace_back(position);
  for (int tokenId : tokenIds) {
    if (tokenId == VocabConstants::getId("UNK_TOKEN")) {
      // We do not produce UNKs while numberizing, hence we are not expected to
      // see them while denumberizing
      FBEXCEPTION("UNK token found while denumberizing");
    } else {
      if (tokenId < 0 || tokenId >= id2token_.size()) {
        FBEXCEPTION(folly::format("Out of range id: {}", tokenId).str());
      }
      // Subword token corresponding to current tokenId
      std::string subtoken = id2token_[tokenId].first;

      if (endsWith(subtoken, VocabConstants::getToken("END_WORD_TOKEN"))) {
        // Adding alignment position for the next word
        alignment.emplace_back(position + 1);
        // Remove the _EOW tag, append to token and add to list of tokens
        token.append(subtoken.substr(
            0,
            subtoken.length() -
                strlen(VocabConstants::getToken("END_WORD_TOKEN"))));
        tokens.emplace_back(token);
        token.clear();
      } else {
        token.append(subtoken);
      }
    }
    ++position;
  }
  // Removing last alignment position added in anticipation of new word
  if (token.empty()) {
    alignment.erase(alignment.end() - 1);
  } else {
    tokens.emplace_back(token);
  }
  return std::make_pair(std::move(tokens), std::move(alignment));
}

size_t BPEVocabProcessor::size() const {
  return VocabConstants::MAX_SPECIAL_TOKENS + tokensAndCounts_.size();
}
} // namespace neural_mt
} // namespace language_technology
} // namespace facebook
