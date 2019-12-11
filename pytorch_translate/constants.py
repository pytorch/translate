#!/usr/bin/env python3

AVERAGED_CHECKPOINT_BEST_FILENAME = "averaged_checkpoint_best.pt"
LAST_CHECKPOINT_FILENAME = "checkpoint_last.pt"

MONOLINGUAL_DATA_IDENTIFIER = "mono"

SEMI_SUPERVISED_TASK = "pytorch_translate_semi_supervised"
KNOWLEDGE_DISTILLATION_TASK = "pytorch_translate_knowledge_distillation"
DENOISING_AUTOENCODER_TASK = "pytorch_translate_denoising_autoencoder"
MULTILINGUAL_TRANSLATION_TASK = "pytorch_translate_multilingual_task"
LATENT_VARIABLE_TASK = "translation_vae"

ARCHS_FOR_CHAR_SOURCE = {
    "char_source",
    "char_source_hybrid",
    "char_source_transformer",
    "char_aware_hybrid",
}
ARCHS_FOR_CHAR_TARGET = {"char_aware_hybrid"}
CHECKPOINT_PATHS_DELIMITER = "|"
