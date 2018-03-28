#pragma once

#include <sys/types.h>
#include <map>
#include <vector>

namespace facebook {
namespace language_technology {
namespace neural_mt {

using TokenToIndexAlignment = std::vector<int>;

using TokenAndCount = std::pair<std::string, std::size_t>;

class BaseVocabProcessor {
 public:
  virtual ~BaseVocabProcessor() = 0;

  virtual std::pair<std::vector<int>, TokenToIndexAlignment> numberize(
      const std::vector<std::string>& tokens) const = 0;

  virtual std::pair<std::vector<std::string>, TokenToIndexAlignment>
  denumberize(const std::vector<int>& tokenIds) const;

  virtual size_t size() const = 0;

  // Some tools (word2vec, KenLM) don't accept numberized input,
  // so we need to tokenize input text, but don't numberize it.
  std::pair<std::vector<std::string>, TokenToIndexAlignment> tokenize(
      const std::vector<std::string>& tokens) const;

  // returns INVALID_ID = -1 if there is no such token in the vocabulary
  virtual int getId(const std::string& token) const = 0;
  std::pair<bool, std::string> getToken(int tokenId) const;
  virtual std::pair<bool, TokenAndCount> getFullToken(int tokenId) const = 0;
  virtual std::pair<bool, TokenAndCount> getTokenWithControl(int tokenId) const;

  // Note, that should hold (sorry for Python syntax):
  // map(getId, tokenize(input_words)) == numberize(input_words)

  void numberizeFile(
      const std::string& inputPath,
      const std::string& outputPath,
      int numThreads = 1);

  void tokenizeFile(
      const std::string& inputPath,
      const std::string& outputPath,
      int numThreads = 1);

  std::vector<int> getProfanityTokenIds() const;
  void setProfanityTokenIds(const std::vector<std::string>& tokens);

  std::vector<TokenAndCount> id2token_;

  // TODO: do profanity token mapping outside of VocabProcessor
  std::vector<int> profanityTokenIds_;
};

inline BaseVocabProcessor::~BaseVocabProcessor() = default;

} // namespace neural_mt
} // namespace language_technology
} // namespace facebook
