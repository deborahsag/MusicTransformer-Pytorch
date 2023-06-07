import argparse


SEPARATOR = "========================="


def parse_pitch_class_entropy_args():
    """
    ----------
    Author: Deborah
    ----------
    Argparse arguments for Overall Pitch Class Histogram Entropy
    ----------
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-midi_root", type=str, help="Folder of midi files to be evaluated.")
    parser.add_argument("-n_resamples", type=int, default=9999, help="Number of resamples for the bootstrap method. Default: 9999.")
    parser.add_argument("-confidence_level", type=float, default=0.95, help="Confidence level for the confidence interval. Default: 0.95.")

    parser.add_argument("--print_each", action="store_true", default=False, help="Print entropy value of each piece.")

    return parser.parse_args()


def print_pitch_class_entropy_args(args):
    """
    ----------
    Author: Deborah
    ----------
    Prints arguments for Overall Pitch Class Histogram Entropy
    ----------
    """

    print(SEPARATOR)
    print("Overall Pitch Class Histogram Entropy")
    print(SEPARATOR)
    print(f"midi_root: {args.midi_root}")
    print(f"n_resamples: {args.n_resamples}")
    print(f"confidence_level: {args.confidence_level}")
    print(SEPARATOR)
    print("")


def parse_pitch_class_consistency_args():
    """
        ----------
        Author: Deborah
        ----------
        Argparse arguments for Pitch Class Consistency Entropy
        ----------
        """

    parser = argparse.ArgumentParser()

    parser.add_argument("-midi_root", type=str, help="Folder of midi files to be evaluated.")
    parser.add_argument("-num_partitions", type=int, default=4, help="Number of partitions for pair to pair comparison.")
    parser.add_argument("-n_resamples", type=int, default=9999, help="Number of resamples for the bootstrap method. Default: 9999.")
    parser.add_argument("-confidence_level", type=float, default=0.95, help="Confidence level for the confidence interval. Default: 0.95.")

    parser.add_argument("--print_each", action="store_true", default=False, help="Print entropy value of each piece.")

    return parser.parse_args()


def print_pitch_class_consistency_args(args):
    """
    ----------
    Author: Deborah
    ----------
    Prints arguments for Pitch Class Consistency Entropy
    ----------
    """

    print(SEPARATOR)
    print("Pitch Class Consistency")
    print(SEPARATOR)
    print(f"midi_root: {args.midi_root}")
    print(f"num_partitions: {args.num_partitions}")
    print(f"n_resamples: {args.n_resamples}")
    print(f"confidence_level: {args.confidence_level}")
    print(SEPARATOR)
    print("")


def parse_mirex_args():
    """
        ----------
        Author: Deborah
        ----------
        Argparse arguments for MIREX
        ----------
        """

    parser = argparse.ArgumentParser()

    parser.add_argument("-midi_root", type=str, help="Folder of midi files to be evaluated.")
    parser.add_argument("-prompt_length", type=int, help="Length of prompt to be fed into the model.")
    parser.add_argument("-continuation_length", type=int, help="Length of the continuation to be produced.")
    parser.add_argument("-num_continuations", type=int, default=4, help="Number of continuations to be chosen for each test. Default 4.")
    parser.add_argument("-num_tests", type=int, default=1, help="Number of evaluations. Pieces are chosen at random. Default 1.")
    parser.add_argument("-seed", type=int, help="Seed for the random sample of continuations.")

    parser.add_argument("-model_weights", type=str, help="Pickled model weights file saved with torch.save and model.state_dict()")
    parser.add_argument("-beam", type=int, default=0, help="Beam search k. 0 for random probability sample and 1 for greedy. Default 0.")

    parser.add_argument("--rpr", action="store_true", help="Use a modified Transformer for Relative Position Representations")
    parser.add_argument("-max_sequence", type=int, default=2048, help="Maximum midi sequence to consider in the model. Default 2048.")
    parser.add_argument("-n_layers", type=int, default=6, help="Number of decoder layers to use. Default 6.")
    parser.add_argument("-num_heads", type=int, default=8, help="Number of heads to use for multi-head attention. Default 8.")
    parser.add_argument("-d_model", type=int, default=512, help="Dimension of the model (output dim of embedding layers, etc. Default 512.)")
    parser.add_argument("-dim_feedforward", type=int, default=1024, help="Dimension of the feedforward layer. Default 1024.")

    return parser.parse_args()


def print_mirex_args(args):
    """
    ----------
    Author: Deborah
    ----------
    Prints arguments for MIREX
    ----------
    """

    print(SEPARATOR)
    print("MIREX")
    print(SEPARATOR)
    print(f"midi_root: {args.midi_root}")
    print(f"prompt_length: {args.prompt_lenght}")
    print(f"continuation_length: {args.continuation_length}")
    print(f"num_continuations: {args.num_continutions}")
    print(f"num_tests: {args.num_tests}")
    print(f"seed: {args.seed}")

    print("---Model arguments---")

    print(f"model_weights: {args.model_weights}")
    print(f"beam: {args.beam}")
    print(f"rpr: {args.rpr}")
    print(f"max_sequence: {args.max_sequence}")
    print(f"n_layers: {args.n_layers}")
    print(f"num_heads: {args.num_heads}")
    print(f"d_model: {args.d_model}")
    print(f"dim_feedforward: {args.dim_feedforward}")

    print(SEPARATOR)
    print("")
