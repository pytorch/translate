#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import multiprocessing
import os
import random
import signal
import torch

from fairseq import distributed_utils
from translate import train
from translate.train import main as single_process_main
from translate import data as translate_data


def main(args):
    # Build vocab from the training corpus. We do this outside of train clones
    # to prevent the clones from having to wait on the master clone building the
    # vocab.
    if args.source_lang is None:
        args.source_lang = 'src'
    if args.target_lang is None:
        args.target_lang = 'tgt'

    args.source_vocab_file = translate_data.build_vocab_if_nonexistent(
        vocab_file=args.source_vocab_file,
        corpus_file=args.train_source_text_file,
        dialect=args.source_lang,
        save_dir=args.save_dir,
        max_vocab_size=args.target_max_vocab_size,
    )
    args.target_vocab_file = translate_data.build_vocab_if_nonexistent(
        vocab_file=args.target_vocab_file,
        corpus_file=args.train_target_text_file,
        dialect=args.target_lang,
        save_dir=args.save_dir,
        max_vocab_size=args.target_max_vocab_size,
        tokens_with_penalty=args.penalized_target_tokens_file,
    )

    # Set distributed training parameters for a single node.
    args.distributed_world_size = torch.cuda.device_count()
    args.distributed_init_method = 'tcp://localhost:{port}'.format(
        port=random.randint(10000, 20000))

    if args.distributed_world_size == 1:
        return single_process_main(args)

    mp = multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(args.distributed_world_size):
        args.distributed_rank = i
        args.device_id = i
        procs.append(mp.Process(target=run, args=(args, error_queue, ), daemon=True))
        procs[i].start()
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


def run(args, error_queue):
    try:
        torch.cuda.set_device(args.device_id)
        args.distributed_rank = distributed_utils.distributed_init(args)
        single_process_main(args)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.distributed_rank, traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        self.children_pids.append(pid)

    def error_listener(self):
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = "\n\n-- Tracebacks above this line can probably be ignored --\n\n"
        msg += original_trace
        raise Exception(msg)


if __name__ == '__main__':
    parser = train.get_parser_with_args()
    args = train.parse_args_and_arch(parser)
    main(args)
