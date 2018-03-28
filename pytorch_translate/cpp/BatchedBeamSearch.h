#pragma once

#include <map>
#include <memory>
#include <vector>

#include <caffe2/core/tensor.h>

#include "DbPredictor.h"
#include "DecoderUtil.h"

namespace pytorch {
namespace translate {

using TensorMap = ::caffe2::Predictor::TensorMap;
using RawTensorMap = std::unordered_map<caffe2::TensorCPU*, DestroyCall>;

class BatchedBeamSearch {
 public:
  BatchedBeamSearch(const std::string& encoderModel,
                    const std::string& decoderStepModel, int beamSize);

  BeamSearchOutput beamSearch(const std::vector<int>& numberizedInput,
                              int maxOutputSeqLen, bool reverseSource);

 private:
  TensorMap prepareInitialNextInputStepMap(
      const std::vector<std::string>& encoderOutputNames,
      const TensorMap& encoderOutputMap, RawTensorMap* trackRawPointers);
  TensorMap prepareNextInputStepMap(
      const std::vector<std::string>& encoderOutputNames,
      const std::vector<std::string>& stepOutputNames,
      TensorMap& encoderOutputMap, const TensorMap& stepOutputMap, int timeStep,
      RawTensorMap* trackRawPointers);

  std::unique_ptr<::caffe2::Workspace> encoder_workspace_;
  std::unique_ptr<::caffe2::Workspace> decoder_workspace_;
  std::unique_ptr<::pytorch::translate::DbPredictor> encoder_;
  std::unique_ptr<::pytorch::translate::DbPredictor> decoderStep_;
  int beamSize_;
};

}  // namespace translate
}  // namespace pytorch
