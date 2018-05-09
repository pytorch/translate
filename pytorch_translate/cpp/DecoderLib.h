#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "BatchedBeamSearch.h"
#include "DecoderUtil.h"
#include "Dictionary.h"

namespace pytorch {
namespace translate {

using TokenToIdAlignment = std::vector<int>;

class NmtDecoder {
 public:
  NmtDecoder(
      int beamSize,
      float maxOutputSeqLenMult,
      int maxOutputSeqLenBias,
      std::shared_ptr<Dictionary> sourceVocab,
      std::shared_ptr<Dictionary> targetVocab,
      const std::string& encoderModel,
      const std::string& decoderStepModel,
      bool reverseSource = true,
      bool stopAtEos = false,
      bool appendEos = false,
      double lengthPenalty = 0);

  TranslationResult translate(std::string input);

  std::vector<TranslationResult> translateNBest(
      std::string input,
      std::size_t nBest);

 private:
  std::vector<Hypothesis> getNBestHypotheses(
      const BeamSearchOutput& beamSearchOutput,
      const std::vector<int>& numberizedInput,
      double lengthPenalty,
      int nBest);

  int beamSize_;
  float maxOutputSeqLenMult_;
  int maxOutputSeqLenBias_;
  std::shared_ptr<Dictionary> sourceVocab_;
  std::shared_ptr<Dictionary> targetVocab_;

  bool reverseSource_;
  bool stopAtEos_;
  bool appendEos_;
  double lengthPenalty_;
  std::unique_ptr<BatchedBeamSearch> batchedBeamSearch_;
};

} // namespace translate
} // namespace pytorch
