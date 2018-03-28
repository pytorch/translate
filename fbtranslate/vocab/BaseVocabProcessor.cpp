#include "language_technology/neural_mt/fbtranslate/vocab/BaseVocabProcessor.h"

#include <functional>

#include <folly/Format.h>
#include <folly/MPMCPipeline.h>
#include <folly/String.h>
#include <cstdio>
#include <thread>

#include "common/base/Exception.h"
#include "common/strings/LineReader.h"
#include "language_technology/neural_mt/fbtranslate/vocab/VocabConstants.h"
namespace {
constexpr int kQueueSizePerThread = 1000;

template <typename T>
void processFile(
    const std::string& inputPath,
    const std::string& outputPath,
    int numThreads,
    std::function<std::vector<T>(const std::vector<std::string>&)> process,
    std::function<void(FILE*, const std::vector<T>&)> writeToFile) {
  if (numThreads <= 0) {
    FBEXCEPTION(
        folly::format("Number of threads should be greater than 0:", numThreads)
            .str());
  }
  LOG(INFO) << folly::format(
      "Numberizing file {} to {} using {} threads",
      inputPath,
      outputPath,
      numThreads);

  // Pipeline consists of 2 queue - of strings and of vector of integers.
  // One thread will read lines from file and write them to the pipeline -
  // - to the first queue.
  // numThreads threads will read from the first queue and process those lines:
  // split into tokens, process them and write results to the second queue.
  // Main thread will read from the pipeline (from the second queue)
  // and will write processed data into the output file.
  // We preserve the order, because MPMCPipeline guarantees it.
  folly::MPMCPipeline<
      std::pair<bool, std::string>,
      std::pair<bool, std::vector<T>>>
      pipeline(
          numThreads * kQueueSizePerThread, numThreads * kQueueSizePerThread);

  std::vector<std::thread> threads;
  threads.reserve(numThreads + 1);
  // These threads will process input lines
  for (size_t i = 0; i < numThreads; ++i) {
    threads.emplace_back([&process, &pipeline]() {
      while (true) {
        std::pair<bool, std::string> flagAndLine;
        auto ticket = pipeline.template blockingReadStage<0>(flagAndLine);
        if (flagAndLine.first) {
          pipeline.template blockingWriteStage<0>(
              ticket, std::make_pair(true, std::vector<T>()));
          break;
        }
        std::vector<std::string> words;
        folly::split(" ", flagAndLine.second, words);
        pipeline.template blockingWriteStage<0>(
            ticket, std::make_pair(false, process(words)));
      }
    });
  }
  // One thread reads from the input file
  threads.emplace_back([numThreads, &inputPath, &pipeline]() {
    const char* line;
    size_t size;
    facebook::strings::LineReader reader(inputPath);
    while (reader.getLine(&line, &size)) {
      pipeline.blockingWrite(std::make_pair(false, std::string(line, size)));
    }
    // sends exit signal to all threads
    for (int i = 0; i < numThreads; ++i) {
      pipeline.blockingWrite(std::make_pair(true, ""));
    }
  });
  // Main thread writes to the output file
  FILE* outputFile = fopen(outputPath.c_str(), "w");
  while (true) {
    std::pair<bool, std::vector<T>> flagsAndProcessedToken;
    pipeline.blockingRead(flagsAndProcessedToken);
    if (flagsAndProcessedToken.first) {
      break;
    }
    writeToFile(outputFile, flagsAndProcessedToken.second);
  }
  fclose(outputFile);

  for (auto& thread : threads) {
    thread.join();
  }
}
} // namespace

namespace facebook {
namespace language_technology {
namespace neural_mt {

void BaseVocabProcessor::numberizeFile(
    const std::string& inputPath,
    const std::string& outputPath,
    int numThreads) {
  processFile<int>(
      inputPath,
      outputPath,
      numThreads,
      [this](const std::vector<std::string>& tokens) -> std::vector<int> {
        return this->numberize(tokens).first;
      },
      [](FILE* outputFile, const std::vector<int>& tokenIds) -> void {
        for (size_t i = 0; i < tokenIds.size(); ++i) {
          if (i > 0) {
            fprintf(outputFile, " ");
          }
          fprintf(outputFile, "%d", tokenIds[i]);
        }
        fprintf(outputFile, "\n");
      });
}

void BaseVocabProcessor::tokenizeFile(
    const std::string& inputPath,
    const std::string& outputPath,
    int numThreads) {
  processFile<std::string>(
      inputPath,
      outputPath,
      numThreads,
      [this](const std::vector<std::string>& tokens)
          -> std::vector<std::string> { return this->tokenize(tokens).first; },
      [](FILE* outputFile, const std::vector<std::string>& tokens) -> void {
        for (size_t i = 0; i < tokens.size(); ++i) {
          if (i > 0) {
            fprintf(outputFile, " ");
          }
          fprintf(outputFile, "%s", tokens[i].c_str());
        }
        fprintf(outputFile, "\n");
      });
}

std::pair<std::vector<std::string>, TokenToIndexAlignment>
BaseVocabProcessor::tokenize(const std::vector<std::string>& tokens) const {
  auto numberized = numberize(tokens);
  std::vector<std::string> tokenized(numberized.first.size());
  std::transform(
      numberized.first.begin(),
      numberized.first.end(),
      tokenized.begin(),
      [this](auto t) {
        auto token = this->getTokenWithControl(t);
        if (!token.first) {
          FBEXCEPTION(folly::format("Unexpected id {}", t).str());
        }
        return token.second.first;
      });

  return std::make_pair(std::move(tokenized), std::move(numberized.second));
}

std::pair<bool, TokenAndCount> BaseVocabProcessor::getTokenWithControl(
    const int tokenId) const {
  return getFullToken(tokenId);
}

std::pair<bool, std::string> BaseVocabProcessor::getToken(
    const int tokenId) const {
  auto res = getFullToken(tokenId);
  return std::make_pair(res.first, std::move(res.second.first));
}

// TODO: do profanity token mapping outside of VocabProcessor
std::vector<int> BaseVocabProcessor::getProfanityTokenIds() const {
  return profanityTokenIds_;
}

void BaseVocabProcessor::setProfanityTokenIds(
    const std::vector<std::string>& tokens) {
  for (auto& token : tokens) {
    auto tokenId = getId(token);
    if (tokenId != VocabConstants::INVALID_ID) {
      profanityTokenIds_.push_back(tokenId);
    }
  }
}

std::pair<std::vector<std::string>, TokenToIndexAlignment>
BaseVocabProcessor::denumberize(const std::vector<int>& tokenIds) const {
  std::vector<std::string> tokens;
  std::string token;
  TokenToIndexAlignment alignment;
  int position = -1;
  bool seenStartToken = false;

  for (int tokenId : tokenIds) {
    ++position;
    if (tokenId == VocabConstants::getId("START_WORD_TOKEN")) {
      seenStartToken = true;
      if (!token.empty()) {
        tokens.emplace_back(token);
      }
      alignment.emplace_back(position);
      token.clear();
    } else if (tokenId == VocabConstants::getId("UNK_TOKEN")) {
      // If an UNK appears in the middle of the word, we want to denumberize
      // the entire word as an UNK.
      token = VocabConstants::getToken("UNK_TOKEN");
      if (seenStartToken == false) {
        alignment.emplace_back(position);
        seenStartToken = true;
      }
    } else {
      if (tokenId < 0 || tokenId >= id2token_.size()) {
        FBEXCEPTION(folly::format("Out of range id: {}", tokenId).str());
      }
      if (seenStartToken == false) {
        alignment.emplace_back(position);
        seenStartToken = true;
      }
      if (token != VocabConstants::getToken("UNK_TOKEN")) {
        token.append(id2token_[tokenId].first);
      }
    }
  }
  if (!token.empty()) {
    tokens.emplace_back(token);
  }
  return std::make_pair(std::move(tokens), std::move(alignment));
}
} // namespace neural_mt
} // namespace language_technology
} // namespace facebook
