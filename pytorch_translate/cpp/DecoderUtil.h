#pragma once

#include <vector>

#include "DbPredictor.h"

namespace pytorch {
namespace translate {

using BeamBestIndices = std::pair<int, int>;
using DestroyCall = caffe2::Blob::DestroyCall;

struct BeamSearchTopHypotheses {
  std::pair<int32_t, int32_t> bestIndices;
  std::vector<std::vector<int>> prevTokens;
  std::vector<std::vector<int>> prevIndexes;
  std::vector<std::vector<int>> topKOutput;
  std::vector<std::vector<int>> alignmentIndexes;
  std::vector<std::vector<float>> outputTokensScore;
  std::vector<std::vector<float>> backAlignmentWeights;
};

struct BeamSearchOutput {
  int numSteps;
  std::vector<std::vector<int>> tokenBeamList;
  std::vector<std::vector<float>> scoreBeamList;
  std::vector<std::vector<int>> prevIndexBeamList;
  std::vector<std::vector<std::vector<float>>> attentionWeightsBeamList;

  BeamSearchOutput() {}

  BeamSearchOutput(
      int ns,
      const std::vector<std::vector<int>>& tbl,
      const std::vector<std::vector<float>>& sbl,
      const std::vector<std::vector<int>>& pibl,
      const std::vector<std::vector<std::vector<float>>>& awbl)
      : numSteps(ns),
        tokenBeamList(tbl),
        scoreBeamList(sbl),
        prevIndexBeamList(pibl),
        attentionWeightsBeamList(awbl) {}
};

struct Hypothesis {
  std::vector<int> numberizedTokenizedOutput;
  std::vector<std::vector<float>> backAlignment;
  float modelScore;
  BeamBestIndices bestIndices;

  Hypothesis(
      std::vector<int> numberizedTokenizedOutput,
      std::vector<std::vector<float>> backAlignment,
      float modelScore,
      BeamBestIndices bestIndices)
      : numberizedTokenizedOutput(numberizedTokenizedOutput),
        backAlignment(backAlignment),
        modelScore(modelScore),
        bestIndices(bestIndices) {}
};

struct BeamHypothesis {
  std::string word;
  int parent;
  float score;
  bool isFinal;
};
using BeamsInfo = std::vector<std::vector<BeamHypothesis>>;

struct TranslationResult {
  std::vector<std::string> denumberized_source;
  std::string translation;
  std::vector<std::string> denumberizedTranslation;
  std::vector<int> numberizedTokenizedInput;
  std::vector<int> numberizedTokenizedOutput;
  std::vector<std::vector<float>> backAlignment;
  float modelScore;
  int numSourceTypes;
  int numTargetTypes;
};

template <typename T>
std::vector<T> tensorToVector1D(const caffe2::TensorCPU& tensor) {
  return std::vector<T>(tensor.data<T>(), tensor.data<T>() + tensor.size());
}

template <typename T>
std::vector<std::vector<T>> tensorToVector2D(const caffe2::TensorCPU& tensor) {
  const auto& dims = tensor.dims();
  const int m = dims[0];
  const int n = dims[1];
  std::vector<std::vector<T>> res(m, std::vector<T>(n));
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      res[i][j] = tensor.data<T>()[i * n + j];
    }
  }
  return res;
}

template <typename T>
std::vector<std::vector<std::vector<T>>> tensorToVector3D(
    const caffe2::TensorCPU& tensor) {
  const auto& dims = tensor.dims();
  const int m = dims[0];
  const int n = dims[1];
  const int l = dims[2];
  std::vector<std::vector<std::vector<T>>> res(
      m, std::vector<std::vector<T>>(n, std::vector<T>(l)));
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < l; ++k) {
        res[i][j][k] = tensor.data<T>()[(i * n + j) * l + k];
      }
    }
  }
  return res;
}

} // namespace translate
} // namespace pytorch
