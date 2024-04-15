import torch
import torch.nn as nn
import os
import random
import pretty_midi
import pickle
import sys
from shutil import copyfile

from third_party.midi_processor.processor import decode_midi, encode_midi
from statistics import mean
from utilities.argument_funcs import parse_generate_args, print_generate_args
from model.music_transformer import MusicTransformer
from dataset.e_piano import create_epiano_datasets, compute_epiano_accuracy, process_midi
from torch.utils.data import DataLoader
from torch.optim import Adam

from utilities.constants import *
from utilities.device import get_device, use_cuda

from miditok import REMI, TokenizerConfig, TokSequence


# REMI+ tokenizer parameters
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "num_velocities": 32,
    "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
    "use_chords": True,
    "use_rests": False,
    "use_tempos": True,
    "use_time_signatures": True,            # REMI+
    "use_programs": True,                   # REMI+
    "one_token_stream_for_programs": True,  # REMI+
    "num_tempos": 32,  # number of tempo bins
    "tempo_range": (40, 250),  # (min, max)
}
config = TokenizerConfig(**TOKENIZER_PARAMS)

# Creates the tokenizer
tokenizer = REMI(config)


def decode_REMIPlus_to_midi(id_list, file_path):
    midi = tokenizer.tokens_to_midi(tokens=id_list)
    midi.dump_midi(file_path)



# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Generates music from a model specified by command line arguments
    ----------
    """
    args = parse_generate_args()
    print_generate_args(args)

    if (args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

    SEED = args.seed if args.seed is not None else random.randrange(sys.maxsize)
    print(f"Setting seed to {SEED}")
    random.seed(SEED)

    os.makedirs(args.output_dir, exist_ok=True)

    # Grabbing dataset if needed
    _, _, dataset = create_epiano_datasets(args.midi_root, args.num_prime, random_seq=False)

    if (args.primer_file is not None):
        f = [args.primer_file]
    else:
        f = random.sample(range(len(dataset)), args.num_primer_files)

    song = 1
    for j in range(args.num_primer_files):
        idx = int(f[j])
        primer, _ = dataset[idx]
        primer = primer.int().to(get_device())

        print("Using primer index:", idx, "(", dataset.data_files[idx], ")")
        with open(dataset.data_files[idx], "rb") as p:
            original = pickle.load(p)
        f_path = os.path.join(args.output_dir, f"original-{idx}.mid")
        decode_REMIPlus_to_midi(original, f_path)
        # decode_midi(original, f"{args.output_dir}/original-{idx}.mid")

        model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                                 d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                                 max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())

        model.load_state_dict(torch.load(args.model_weights))

        f_path = os.path.join(args.output_dir, f"primer-{idx}.mid")
        decode_REMIPlus_to_midi(primer[:args.num_prime].tolist(), f_path)
        # decode_midi(primer[:args.num_prime].tolist(), f_path)

        for i in range(args.num_samples):
            print(f"Generating song {song}/{args.num_primer_files * args.num_samples}")
            # GENERATION
            model.eval()
            with torch.set_grad_enabled(False):
                if (args.beam > 0):
                    print("BEAM:", args.beam)
                    beam_seq, _ = model.generate(primer[:args.num_prime], args.target_seq_length, beam=args.beam)

                    f_path = os.path.join(args.output_dir, f"beam-{idx}-{i}.mid")
                    decode_REMIPlus_to_midi(beam_seq[0].tolist(), f_path)
                    # decode_midi(beam_seq[0].tolist(), f_path)
                else:
                    print("RAND DIST")
                    rand_seq, _ = model.generate(primer[:args.num_prime], args.target_seq_length, beam=0)

                    f_path = os.path.join(args.output_dir, f"rand-{idx}-{i}.mid")
                    decode_REMIPlus_to_midi(rand_seq[0].tolist(), f_path)
                    # decode_midi(rand_seq[0].tolist(), f_path)

            song += 1
            print()


if __name__ == "__main__":
    main()