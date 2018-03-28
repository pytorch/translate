#include "language_technology/neural_mt/fbtranslate/vocab/Loader.h"

#include <fstream>
#include <string>

// TODO: include all vocab processors once thrift dependencies removed from them
#include "language_technology/neural_mt/fbtranslate/vocab/BPEVocabProcessor.h"
#include "language_technology/neural_mt/fbtranslate/vocab/BaseVocabProcessor.h"
// #include
// "language_technology/neural_mt/fbtranslate/vocab/CharNgramVocabProcessor.h"
// #include
// "language_technology/neural_mt/fbtranslate/vocab/MorfessorVocabProcessor.h"
#include "language_technology/neural_mt/fbtranslate/vocab/VocabConstants.h"
#include "language_technology/neural_mt/fbtranslate/vocab/WordVocabProcessor.h"
// #include
// "language_technology/neural_mt/fbtranslate/vocab/WordpieceVocabProcessor.h"

namespace facebook {
namespace language_technology {
namespace neural_mt {

std::shared_ptr<BaseVocabProcessor> loadVocabProcessorFromFile(
    std::string path) {
  std::ifstream vocabFile(path);

  // Read vocab type from top of file
  std::string vocabType;
  std::getline(vocabFile, vocabType);

  size_t vocabSize;
  vocabFile >> vocabSize; // Read vocab size
  vocabFile.get(); // Skip newline char

  // Build TokenAndCounts list from the (token, token count) list
  std::vector<TokenAndCount> tokensAndCounts;
  for (size_t i = 0; i < vocabSize; i++) {
    std::string token;
    size_t count;
    // assumes delimiter between token and count is whitespace
    vocabFile >> token >> count;
    TokenAndCount w = {token, count};
    tokensAndCounts.emplace_back(w);
  }

  // TODO: switch on vocab type to find correct VocabProcessor to use
  if (vocabType == VocabConstants::WORD_VOCAB_TYPE) {
    return std::make_unique<WordVocabProcessor>(tokensAndCounts);
  } else if (vocabType == VocabConstants::BPE_VOCAB_TYPE) {
    return std::make_unique<BPEVocabProcessor>(tokensAndCounts);
  } else {
    throw std::invalid_argument("Unsupported vocab type.");
  }
}
} // namespace neural_mt
} // namespace language_technology
} // namespace facebook
