import argparse
import json
import os
import pickle
from copy import deepcopy

import pretty_midi

from third_party.midi_processor.processor import encode_midi

pitch_shift = [-3, -2, -1, 0, 1, 2, 3]
tempo_shift = [0.95, 0.975, 1, 1.025, 1.05]

JSON_FILE = "maestro-v2.0.0.json"


def augment_midi(midi, pitch, tempo):
    """
    ----------
    Authors:  Gabriel Souza, Deborah Guimarães
    ----------
    Data augmentation using the pretty_midi module.
    ----------
    """

    print(f"Doing for {midi}")
    dir_split, extension_split = midi.rfind("/") + 1, midi.rfind(".midi") if ".midi" in midi else midi.rfind(".MIDI")
    filepath, filename, extension = midi[:dir_split], midi[dir_split:extension_split], midi[extension_split:]

    mid = pretty_midi.PrettyMIDI(midi)

    mid.remove_invalid_notes()
    mid.instruments = [instr for instr in mid.instruments if not instr.is_drum]     # remove percussion

    new_mid = deepcopy(mid)
    for instr in new_mid.instruments:
        for note in instr.notes:
            note.pitch += pitch
            if note.pitch > 127:
                note.pitch -= 12
            elif note.pitch < 0:
                note.pitch += 12
            note.start, note.end = note.start * tempo, note.end * tempo

    print(f"Writing {filepath}Aug-{filename}-{pitch}-{tempo}{extension}")
    new_mid.write(f"{filepath}Aug-{filename}-{pitch}-{tempo}{extension}")

    return True



def prep_maestro_midi_aug(maestro_root, output_dir):
    """
    ----------
    Author: Damon Gwinn, alterations: Gabriel Souza, Deborah Guimarães
    ----------
    Pre-processes the maestro dataset, putting processed midi data (train, eval, test) into the
    given output folder, while making the data augmentation for the train data
    ----------
    """

    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(val_dir, exist_ok=True)
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    maestro_json_file = os.path.join(maestro_root, JSON_FILE)
    if not os.path.isfile(maestro_json_file):
        print("ERROR: Could not find file:", maestro_json_file)
        return False

    maestro_json = json.load(open(maestro_json_file, "r"))
    print("Found", len(maestro_json), "pieces")
    print("Preprocessing...")

    total_count = 0
    train_count = 0
    val_count   = 0
    test_count  = 0

    for piece in maestro_json:
        mid         = os.path.join(maestro_root, piece["midi_filename"])
        split_type  = piece["split"]
        f_name      = mid.split("/")[-1] + ".pickle"

        if split_type == "train":
            o_file = os.path.join(train_dir, f_name)
            train_count += 1
        elif split_type == "validation":
            o_file = os.path.join(val_dir, f_name)
            val_count += 1
        elif split_type == "test":
            o_file = os.path.join(test_dir, f_name)
            test_count += 1
        else:
            print("ERROR: Unrecognized split type:", split_type)
            return False

        prepped = encode_midi(mid)

        o_stream = open(o_file, "wb")
        pickle.dump(prepped, o_stream)
        o_stream.close()
        
        if split_type == "train":
            for pitch in pitch_shift:
                for tempo in tempo_shift:
                    if (pitch == 0 and tempo == 1) or os.path.exists(f"{output_dir}/train/Aug-{piece['midi_filename']}-{pitch}-{tempo}.pickle"):
                        continue

                    augment_midi(mid, pitch, tempo)

                    piece_name = piece["midi_filename"]
                    name_aug = f"Aug-{piece_name}-{pitch}-{tempo}.midi"
                    mid_aug    = os.path.join(maestro_root, name_aug)
                    f_name_aug = mid_aug.split("/")[-1] + ".pickle"
                    o_file_aug = os.path.join(train_dir, f_name_aug)
                    prepped_aug = encode_midi(mid_aug)
                    o_stream_aug = open(o_file_aug, "wb")
                    pickle.dump(prepped_aug, o_stream_aug)
                    o_stream_aug.close()

        total_count += 1
        if total_count % 50 == 0:
            print(total_count, "/", len(maestro_json))

    print("Num Train:", train_count)
    print("Num Val:", val_count)
    print("Num Test:", test_count)
    return True


def parse_args():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Parses arguments for preprocess_midi using argparse
    ----------
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("root", type=str, help="Root folder for the Maestro dataset or for custom data.")
    parser.add_argument("-output_dir", type=str, default="./dataset/e_piano", help="Output folder to put the preprocessed midi into.")
    # parser.add_argument("--custom_dataset", action="store_true", help="Whether or not the specified root folder contains custom data.")

    return parser.parse_args()


def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Preprocesses maestro and saved midi to specified output folder.
    ----------
    """

    args            = parse_args()
    root            = args.root
    output_dir      = args.output_dir

    print("Preprocessing midi files and saving to", output_dir)
    prep_maestro_midi_aug(root, output_dir)
    print("Done!")
    print("")


if __name__ == "__main__":
    main()
