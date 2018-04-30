#include "common/init/Init.h"

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <caffe2/core/flags.h>

#include "DecoderLib.h"

namespace pyt = pytorch::translate;

CAFFE2_DEFINE_string(encoder_model, "", "Encoder model path");
CAFFE2_DEFINE_string(decoder_step_model, "", "Decoder step model path");
CAFFE2_DEFINE_double(
    max_out_seq_len_mult,
    -1,
    "determines max num tokens in translation based on num tokens in input"
    "max_out_tokens = "
    "input_tokens * max_out_seq_len_mult + max_out_seq_len_bias");
CAFFE2_DEFINE_int(
    max_out_seq_len_bias,
    -1,
    "determines max num tokens in translation based on num tokens in input"
    "max_out_tokens = "
    "input_tokens * max_out_seq_len_mult + max_out_seq_len_bias");
CAFFE2_DEFINE_int(beam_size, -1, "Beam size");
CAFFE2_DEFINE_string(source_vocab_path, "", "Source vocab file");
CAFFE2_DEFINE_string(target_vocab_path, "", "Target vocab file");
CAFFE2_DEFINE_bool(
    reverse_source,
    true,
    "reverse source sentence before encoding");
CAFFE2_DEFINE_bool(
    stop_at_eos,
    false,
    "do not consider sequences containing a non-final EOS token");
CAFFE2_DEFINE_bool(
    append_eos_to_source,
    false,
    "appending EOS token to source sentence.");
CAFFE2_DEFINE_double(
    length_penalty,
    0,
    "hypothesis score divided by (numwords ^ length_penalty)");

int main(int argc, char** argv) {
  facebook::initFacebook(&argc, &argv);
  std::shared_ptr<pyt::Dictionary> sourceVocab =
      std::make_shared<pyt::Dictionary>(FLAGS_source_vocab_path);
  std::shared_ptr<pyt::Dictionary> targetVocab =
      std::make_shared<pyt::Dictionary>(FLAGS_target_vocab_path);
  std::shared_ptr<pyt::NmtDecoder> decoder = std::make_shared<pyt::NmtDecoder>(
      FLAGS_beam_size,
      FLAGS_max_out_seq_len_mult,
      FLAGS_max_out_seq_len_bias,
      std::move(sourceVocab),
      std::move(targetVocab),
      FLAGS_encoder_model,
      FLAGS_decoder_step_model,
      FLAGS_reverse_source,
      FLAGS_stop_at_eos,
      FLAGS_append_eos_to_source,
      FLAGS_length_penalty);

  if (decoder == nullptr) {
    LOG(FATAL) << "failed to load decoder";
  }

  LOG(INFO) << "Ready to translate";

  for (std::string line; std::getline(std::cin, line);) {
    pyt::TranslationResult translationResult;
    translationResult = decoder->translate(line);
    std::cout << translationResult.translation << "\n";
  }
}
