#include "Dictionary.h"

#include <errno.h>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace pytorch {
namespace translate {

Dictionary::Dictionary(
    const std::string& file_path,
    const std::string& padSymbol,
    const std::string& eosSymbol,
    const std::string& unkSymbol)
    : padSymbol_(padSymbol), eosSymbol_(eosSymbol), unkSymbol_(unkSymbol) {
  idToToken_.reserve(kMaxSpecialTokens);
  tokenToId_.reserve(kMaxSpecialTokens);

  padId_ = addToken(padSymbol_);
  // Insert a dummy token for the go symbol for backward compatibility reasons.
  goId_ = addToken(kReservedSymbol);
  eosId_ = addToken(eosSymbol_);
  unkId_ = addToken(unkSymbol_);

  while (idToToken_.size() < kMaxSpecialTokens) {
    addToken(kReservedSymbol);
  }

  std::ifstream input(file_path);
  if (input.fail()) {
    std::ostringstream errMessage;
    errMessage << "Could not open file " << file_path << " - "
               << std::strerror(errno) << std::endl;
    throw std::invalid_argument(errMessage.str());
  }

  int line_num = 0;
  for (std::string line; std::getline(input, line); ++line_num) {
    // Split on the last space, in accordance with fairseq's dictionary.py.
    std::size_t split_index = line.rfind(" ");
    if (split_index == std::string::npos) {
      std::ostringstream errMessage;
      errMessage << "Invalid format in file " << file_path << " - line "
                 << line_num << " does not contain a space." << std::endl;
      throw std::invalid_argument(errMessage.str());
    }

    std::string token = line.substr(0, split_index);
    std::string countStr = line.substr(split_index + 1);
    try {
      // We don't currently use the count, so it's not stored. Update addToken()
      // to store the count if it becomes useful in the future.
      std::stoi(countStr);
    } catch (const std::invalid_argument& e) {
      std::ostringstream errMessage;
      errMessage << "Invalid format in file " << file_path << " line "
                 << line_num << " - '" << countStr
                 << "' can't be cast as an int." << std::endl;
      throw std::invalid_argument(errMessage.str());
    }
    addToken(token);
  }
}

int Dictionary::addToken(const std::string& token) {
  idToToken_.push_back(token);
  const int id = idToToken_.size() - 1;
  tokenToId_[token] = id;
  return id;
}

// static
std::vector<std::string> Dictionary::tokenize(const std::string& line) {
  // Splits string by whitespace. istream_iterator skips whitespace by
  // default.
  std::vector<std::string> output;
  std::istringstream stringStream(line, std::ios_base::out);
  copy(
      std::istream_iterator<std::string>(stringStream),
      std::istream_iterator<std::string>(),
      back_inserter(output));
  return output;
}

std::vector<int> Dictionary::numberize(
    const std::vector<std::string>& tokens) const {
  std::vector<int> ids;
  ids.reserve(tokens.size());
  for (const std::string& token : tokens) {
    const auto iter = tokenToId_.find(token);
    if (iter == tokenToId_.end()) {
      ids.push_back(unkId_);
      continue;
    }
    ids.push_back(iter->second);
  }
  return ids;
}

std::vector<std::string> Dictionary::denumberize(
    const std::vector<int>& ids) const {
  std::vector<std::string> tokens;
  tokens.reserve(ids.size());
  for (const int id : ids) {
    if (id < 0 || id >= idToToken_.size()) {
      tokens.push_back(unkSymbol_);
      continue;
    }
    tokens.push_back(idToToken_[id]);
  }
  return tokens;
}

} // namespace translate
} // namespace pytorch
