#include "DecoderLib.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>

#include <caffe2/core/common.h>
#include <caffe2/core/logging.h>

#include "DecoderUtil.h"

namespace pytorch {
namespace translate {

NmtDecoder::NmtDecoder(
    int beamSize,
    float maxOutputSeqLenMult,
    int maxOutputSeqLenBias,
    std::shared_ptr<Dictionary> sourceVocab,
    std::shared_ptr<Dictionary> targetVocab,
    const std::string& encoderModel,
    const std::string& decoderStepModel,
    bool reverseSource,
    bool stopAtEos,
    bool appendEos,
    double lengthPenalty)
    : beamSize_(beamSize),
      maxOutputSeqLenMult_(maxOutputSeqLenMult),
      maxOutputSeqLenBias_(maxOutputSeqLenBias),
      sourceVocab_(std::move(sourceVocab)),
      targetVocab_(std::move(targetVocab)),
      reverseSource_(reverseSource),
      stopAtEos_(stopAtEos),
      appendEos_(appendEos),
      lengthPenalty_(lengthPenalty) {
  batchedBeamSearch_ = caffe2::make_unique<BatchedBeamSearch>(
      encoderModel, decoderStepModel, beamSize_);
  LOG(INFO) << "C2 NMT Decoder is initialized";
}

std::vector<Hypothesis> NmtDecoder::getNBestHypotheses(
    const BeamSearchOutput& beamSearchOutput,
    const std::vector<int>& numberizedInput,
    const double lengthPenalty,
    const int nBest) {
  // the pair of int represents the ending position of a hypothesis in
  // the beam search grid. The first index corresponds to the length
  // (column index), the second index corresponds to the position in the
  // beam (row index). The float is the score of the hypothesis.
  std::priority_queue<std::pair<float, std::pair<int, int>>> endStates;

  std::vector<bool> prevHypoIsFinished(beamSize_, false);
  std::vector<bool> currentHypoIsFinished(beamSize_, false);
  for (std::size_t lengthIndex = 1; lengthIndex < beamSearchOutput.numSteps + 1;
       ++lengthIndex) {
    currentHypoIsFinished.assign(beamSize_, false);
    for (std::size_t hypIndex = 0; hypIndex < beamSize_; ++hypIndex) {
      currentHypoIsFinished[hypIndex] =
          prevHypoIsFinished[beamSearchOutput
                                 .prevIndexBeamList[lengthIndex][hypIndex]];
      if (!currentHypoIsFinished[hypIndex] &&
          (beamSearchOutput.tokenBeamList[lengthIndex][hypIndex] == kEosId ||
           lengthIndex == beamSearchOutput.numSteps)) {
        if (stopAtEos_) {
          currentHypoIsFinished[hypIndex] = true;
        }
        float score = beamSearchOutput.scoreBeamList[lengthIndex][hypIndex] /
            pow(lengthIndex, lengthPenalty);
        if (endStates.size() == nBest) {
          if (score > -endStates.top().first) {
            endStates.pop();
            endStates.emplace(
                std::make_pair(-score, std::make_pair(lengthIndex, hypIndex)));
          } else {
            continue;
          }
        } else {
          endStates.emplace(
              std::make_pair(-score, std::make_pair(lengthIndex, hypIndex)));
        }
      }
    }
    prevHypoIsFinished.swap(currentHypoIsFinished);
  }

  std::vector<Hypothesis> results;
  const auto numEndStates = endStates.size();
  for (std::size_t h = 0; h < numEndStates; ++h) {
    auto scoredIndices = endStates.top();
    float modelScore = -scoredIndices.first;
    auto indices = scoredIndices.second;
    endStates.pop();

    std::vector<float> outputStepToScore(indices.first + 1);
    std::vector<std::size_t> outputStepToHypIndex(indices.first + 1);
    std::vector<int> output;
    std::vector<std::vector<float>> backAlignmentWeights;
    {
      auto stepIndex = indices.first;
      auto hypIndex = indices.second;
      if (beamSearchOutput.attentionWeightsBeamList.empty()) {
        throw std::runtime_error("Empty attention weights");
      }
      while (stepIndex > 0) {
        outputStepToScore[stepIndex] =
            beamSearchOutput.scoreBeamList[stepIndex][hypIndex];
        outputStepToHypIndex[stepIndex] = hypIndex;
        output.emplace_back(
            beamSearchOutput.tokenBeamList[stepIndex][hypIndex]);
        std::vector<float> weights;
        if (beamSearchOutput.attentionWeightsBeamList[stepIndex].empty()) {
          throw std::runtime_error("Empty attention weights");
        }
        for (std::size_t sourceIndex = 0; sourceIndex < numberizedInput.size();
             ++sourceIndex) {
          if (beamSearchOutput.attentionWeightsBeamList[stepIndex][hypIndex]
                  .empty()) {
            throw std::runtime_error("Empty attention weights");
          }
          weights.emplace_back(
              beamSearchOutput
                  .attentionWeightsBeamList[stepIndex][hypIndex][sourceIndex]);
        }
        backAlignmentWeights.emplace_back(weights);
        hypIndex = beamSearchOutput.prevIndexBeamList[stepIndex][hypIndex];
        stepIndex--;
      }
    }
    std::reverse(output.begin(), output.end());
    std::reverse(backAlignmentWeights.begin(), backAlignmentWeights.end());
    for (auto& weights : backAlignmentWeights) {
      if (reverseSource_) {
        std::reverse(weights.begin(), weights.end());
      }
      weights.resize(numberizedInput.size());
    }
    auto findEos = std::find(output.begin(), output.end(), kEosId);
    int findEosIndex = findEos - output.begin();
    output.resize(findEosIndex);
    backAlignmentWeights.resize(findEosIndex);

    float bestEosScore = NAN;
    std::vector<int> stepToBestEosHypIndex(beamSearchOutput.numSteps + 1);
    for (std::size_t i = 1; i < beamSearchOutput.numSteps + 1; ++i) {
      stepToBestEosHypIndex[i] = -1;
      for (std::size_t hypIndex = 0; hypIndex < beamSize_; ++hypIndex) {
        if (beamSearchOutput.tokenBeamList[i][hypIndex] == kEosId &&
            (std::isnan(bestEosScore) ||
             bestEosScore < beamSearchOutput.scoreBeamList[i][hypIndex])) {
          bestEosScore = beamSearchOutput.scoreBeamList[i][hypIndex];
          stepToBestEosHypIndex[i] = hypIndex;
        }
      }
    }

    results.emplace_back(
        std::move(output),
        std::move(backAlignmentWeights),
        modelScore,
        std::move(indices));
  }

  std::reverse(results.begin(), results.end());
  return results;
}

TranslationResult NmtDecoder::translate(std::string input) {
  const auto& results = translateNBest(std::move(input), 1);
  if (results.empty()) {
    LOG(INFO) << "Empty n-best, returning empty translation";
    return TranslationResult();
  }
  return results[0];
}

std::vector<TranslationResult> NmtDecoder::translateNBest(
    std::string input,
    std::size_t nBest) {
  std::vector<TranslationResult> results;

  std::vector<std::string> tokenizedSource = sourceVocab_->tokenize(input);
  std::vector<int> numberizedTokens = sourceVocab_->numberize(tokenizedSource);

  if (numberizedTokens.empty()) {
    LOG(INFO) << "Empty numberized input, returning empty translation";
    return results;
  }

  if (appendEos_) {
    numberizedTokens.emplace_back(kEosId);
  }

  auto maxOutputSeqLen = static_cast<int>(std::ceil(
      numberizedTokens.size() * maxOutputSeqLenMult_ + maxOutputSeqLenBias_));
  std::vector<Hypothesis> nbestHypotheses;
  BeamSearchOutput beamSearchOutput;
  beamSearchOutput = batchedBeamSearch_->beamSearch(
      numberizedTokens, maxOutputSeqLen, reverseSource_);
  nbestHypotheses = getNBestHypotheses(
      beamSearchOutput, numberizedTokens, lengthPenalty_, nBest);

  for (auto& hypothesis : nbestHypotheses) {
    TranslationResult result;

    result.denumberized_source = tokenizedSource;
    result.numberizedTokenizedOutput =
        std::move(hypothesis.numberizedTokenizedOutput);
    result.backAlignment = std::move(hypothesis.backAlignment);
    result.modelScore = hypothesis.modelScore;
    result.numberizedTokenizedInput = numberizedTokens;

    std::vector<std::string> denumberizedTranslation =
        targetVocab_->denumberize(result.numberizedTokenizedOutput);
    std::string translation = caffe2::Join(" ", denumberizedTranslation);
    result.denumberizedTranslation = std::move(denumberizedTranslation);
    result.translation = std::move(translation);

    result.numSourceTypes = std::unordered_set<int>(
                                result.numberizedTokenizedInput.begin(),
                                result.numberizedTokenizedInput.end())
                                .size();
    result.numTargetTypes = std::unordered_set<int>(
                                result.numberizedTokenizedOutput.begin(),
                                result.numberizedTokenizedOutput.end())
                                .size();

    results.emplace_back(result);
  }
  return results;
}

} // namespace translate
} // namespace pytorch
