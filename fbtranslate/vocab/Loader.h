#pragma once

#include <folly/dynamic.h>
#include <fstream>
#include <memory>

#include "language_technology/neural_mt/fbtranslate/vocab/BaseVocabProcessor.h"

namespace facebook {
namespace language_technology {
namespace neural_mt {

std::shared_ptr<BaseVocabProcessor> loadVocabProcessorFromFile(
    std::string path);

} // namespace neural_mt
} // namespace language_technology
} // namespace facebook
