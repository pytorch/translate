// Simple dictionary paralleling dictionary.py.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace pytorch {
namespace translate {

// TODO: Consider using Cython to keep these in sync with Python.
// These should be kept in sync with translate/dictionary.py
constexpr char kPadSymbol[] = "<pad>";
constexpr char kEosSymbol[] = "</s>";
constexpr char kUnkSymbol[] = "<unk>";
constexpr char kReservedSymbol[] = "<reserved>";
// These should be kept in sync with translate/vocab_constants.py
constexpr int kMaxSpecialTokens = 100;
constexpr int kPadId = 0;
constexpr int kGoId = 1;
constexpr int kEosId = 2;
constexpr int kUnkId = 3;

class Dictionary {
 public:
  explicit Dictionary(
      const std::string& file_path,
      const std::string& padSymbol = kPadSymbol,
      const std::string& eosSymbol = kEosSymbol,
      const std::string& unkSymbol = kUnkSymbol);

  // Disable copy constructor and copy assignment operator.
  Dictionary(const Dictionary&) = delete;
  Dictionary& operator=(const Dictionary&) = delete;

  static std::vector<std::string> tokenize(const std::string& line);

  std::vector<int> numberize(const std::vector<std::string>& tokens) const;

  std::vector<std::string> denumberize(const std::vector<int>& ids) const;

  const std::string& padSymbol() const {
    return padSymbol_;
  }
  const std::string& eosSymbol() const {
    return eosSymbol_;
  }
  const std::string& unkSymbol() const {
    return unkSymbol_;
  }

  int padId() const {
    return padId_;
  }
  int eosId() const {
    return eosId_;
  }
  int unkId() const {
    return unkId_;
  }
  int size() const {
    return idToToken_.size();
  }

 private:
  int addToken(const std::string& token);

  std::vector<std::string> idToToken_;
  std::unordered_map<std::string, int> tokenToId_;

  const std::string padSymbol_;
  const std::string eosSymbol_;
  const std::string unkSymbol_;
  int padId_;
  int goId_;
  int eosId_;
  int unkId_;
};

} // namespace translate
} // namespace pytorch
