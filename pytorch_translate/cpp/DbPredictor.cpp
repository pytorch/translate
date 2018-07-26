#include "DbPredictor.h"

#include <caffe2/predictor/predictor_utils.h>

namespace pytorch {
namespace translate {

DbPredictor::DbPredictor(const std::string& dbPath, ::caffe2::Workspace* parent)
    : Predictor(
          *::caffe2::predictor_utils::runGlobalInitialization(
              caffe2::make_unique<::caffe2::db::DBReader>("minidb", dbPath),
              parent),
          parent,
          false) {}

} // namespace translate
} // namespace pytorch
