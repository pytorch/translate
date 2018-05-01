// A version of Caffe2 Predictor that can load a model from db file path.

#pragma once

#include <string>

#include <caffe2/core/predictor.h>
#include <caffe2/core/workspace.h>

namespace pytorch {
namespace translate {

class DbPredictor : public ::caffe2::Predictor {
 public:
  DbPredictor(const std::string& dbPath, ::caffe2::Workspace* parent);
  ~DbPredictor() {}
};

}  // namespace translate
}  // namespace pytorch
