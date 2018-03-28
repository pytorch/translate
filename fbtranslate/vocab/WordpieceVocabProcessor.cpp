#include "language_technology/neural_mt/fbtranslate/vocab/WordpieceVocabProcessor.h"

#include <folly/Format.h>

#include "common/base/Exception.h"
#include "common/strings/icu/ICUString.h"
#include "language_technology/neural_mt/fbtranslate/vocab/VocabConstants.h"
#include "language_technology/yoda/YodaTagUtil.h"

#include <vector>

namespace facebook {
namespace language_technology {
namespace neural_mt {

WordpieceVocabProcessor::WordpieceVocabProcessor(WordpieceVocab wordpieceVocab)
    : wordpieceVocab_(std::move(wordpieceVocab)) {
  auto yoda_tags_set = yoda::getYodaTags();
  std::vector<std::string> yoda_tags(
      yoda_tags_set.begin(), yoda_tags_set.end());

  if (VocabConstants::SPECIAL_TOKENS().size() + yoda_tags.size() >
      VocabConstants::MAX_SPECIAL_TOKENS) {
    FBEXCEPTION("There are too many special tokens.");
  }
  token2id_.reserve(
      VocabConstants::SPECIAL_TOKENS().size() + yoda_tags.size() +
      wordpieceVocab_.tokens.size());

  Word unk_word;
  unk_word.token = VocabConstants::getToken("UNDEFINED_TOKEN");
  unk_word.count = 0;
  id2token_.resize(
      VocabConstants::MAX_SPECIAL_TOKENS + wordpieceVocab_.tokens.size(),
      unk_word);

  for (size_t i = 0; i < VocabConstants::SPECIAL_TOKENS().size(); ++i) {
    auto ret = token2id_.insert(
        std::make_pair(VocabConstants::SPECIAL_TOKENS()[i], i));
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
    auto ret = token2id_.insert(std::make_pair(
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

  for (size_t i = 0; i < wordpieceVocab_.tokens.size(); ++i) {
    // Because the yoda_tags may be in the words already, continue if this word
    // is a yoda_tag.
    if (yoda_tags_set.find(wordpieceVocab_.tokens[i].token) !=
        yoda_tags_set.end()) {
      continue;
    }

    auto ret = token2id_.insert(std::make_pair(
        wordpieceVocab_.tokens[i].token,
        i + VocabConstants::MAX_SPECIAL_TOKENS));
    if (ret.second == false) {
      FBEXCEPTION(folly::format(
                      "Non-special element {} already exists",
                      wordpieceVocab_.tokens[i].token)
                      .str());
    }

    id2token_[i + VocabConstants::MAX_SPECIAL_TOKENS] =
        wordpieceVocab_.tokens[i];
  }
}

int WordpieceVocabProcessor::getId(const std::string& token) const {
  auto found = token2id_.find(token);
  return (found != token2id_.end()) ? found->second
                                    : VocabConstants::INVALID_ID;
}

std::pair<bool, Word> WordpieceVocabProcessor::getFullToken(int tokenId) const {
  if (tokenId < 0 || tokenId >= id2token_.size()) {
    return std::make_pair(false, Word());
  }

  return std::make_pair(true, id2token_[tokenId]);
}

std::pair<std::vector<int>, TokenToIndexAlignment>
WordpieceVocabProcessor::numberize(
    const std::vector<std::string>& tokens) const {
  std::vector<int> res;
  TokenToIndexAlignment alignment;
  for (const auto& token : tokens) {
    // If the entire token exists in the corpus, skip Wordpiece and add it
    // directly.
    if (getId(token) != VocabConstants::INVALID_ID) {
      alignment.emplace_back(res.size());
      res.emplace_back(VocabConstants::getId("START_WORD_TOKEN"));
      res.emplace_back(getId(token));
      continue;
    }

    std::vector<std::string> encoded_token;
    encoded_token.emplace_back(VocabConstants::getToken("START_WORD_TOKEN"));

    icu::UnicodeString utoken = icu::UnicodeString::fromUTF8(token);

    // For the word we're currently encoding: create a vector of character
    // tokens. If a character does not exist as a token -- it does not exist
    // as a subset of any token -- so we will have to encode it as an UNK.
    for (int i = 0; i < utoken.length(); ++i) {
      icu::UnicodeString ucur_character(utoken[i]);
      std::string cur_character;
      ucur_character.toUTF8String(cur_character);
      if (getId(cur_character) != VocabConstants::INVALID_ID) {
        encoded_token.emplace_back(cur_character);
      } else {
        encoded_token.emplace_back(VocabConstants::getToken("UNK_TOKEN"));
      }
    }

    while (true) {
      int cur_id;
      int best_id = size() + 1;
      int best_ind = -1;

      // Find the most frequent bigram of tokens. Tokens with lower IDs are
      // more (or equally) frequent than tokens with higher IDs.
      for (int i = 0; i < encoded_token.size() - 1; ++i) {
        if (encoded_token[i] == VocabConstants::getToken("UNK_TOKEN") ||
            encoded_token[i + 1] == VocabConstants::getToken("UNK_TOKEN")) {
          continue;
        }

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

size_t WordpieceVocabProcessor::size() const {
  return VocabConstants::MAX_SPECIAL_TOKENS + wordpieceVocab_.tokens.size();
}
} // namespace neural_mt
} // namespace language_technology
} // namespace facebook
