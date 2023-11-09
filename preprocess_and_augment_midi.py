"""
Observacoes

Pode tambem verificar no preprocess se a musica ja foi copiada pra evitar copiar de novo

test.json
train: 8
test: 4
validation: 7

"""

import argparse
import json
import os
import pickle
from copy import deepcopy

import pretty_midi

from third_party.midi_processor.processor import encode_midi

pitch_shift = [-3, -2, -1, 0, 1, 2, 3]
time_stretch = [0.95, 0.975, 1, 1.025, 1.05]

JSON_FILE = "test.json"


def augment_midi(midi_path, pitch, tempo):
    """
    ----------
    Authors:  Gabriel Souza, Deborah Guimarães
    ----------
    Data augmentation using the pretty_midi module.
    ----------
    """

    print(f"Doing for {midi_path}")
    dir_split, extension_split = midi_path.rfind("/") + 1, midi_path.rfind(".midi") if ".midi" in midi_path else midi_path.rfind(".MIDI")
    filepath, filename, extension = midi_path[:dir_split], midi_path[dir_split:extension_split], midi_path[extension_split:]

    midi = pretty_midi.PrettyMIDI(midi_path)

    midi.remove_invalid_notes()
    midi.instruments = [instr for instr in midi.instruments if not instr.is_drum]     # remove percussion

    new_midi = deepcopy(midi)
    for instr in new_midi.instruments:
        for note in instr.notes:
            note.pitch += pitch
            if note.pitch > 127:
                note.pitch -= 12
            elif note.pitch < 0:
                note.pitch += 12
            note.start, note.end = note.start * tempo, note.end * tempo

    print(f"Writing {filepath}Aug-{filename}-{pitch}-{tempo}{extension}")
    new_midi.write(f"{filepath}Aug-{filename}-{pitch}-{tempo}{extension}")

    return True


def prep_maestro_midi_aug(maestro_root, output_dir):
    """
    ----------
    Author: Damon Gwinn, Deborah Guimarães, Gabriel Souza
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
    aug_count   = 0
    skip_count  = 0

    for piece in maestro_json:
        midi_path         = os.path.join(maestro_root, piece["midi_filename"])
        split_type  = piece["split"]
        f_name      = midi_path.split("/")[-1] + ".pickle"

        if split_type == "train":
            o_file = os.path.join(train_dir, f_name)
            # train_count += 1      # comment this line to prevent counting non-augmented data twice
        elif split_type == "validation":
            o_file = os.path.join(val_dir, f_name)
            val_count += 1
        elif split_type == "test":
            o_file = os.path.join(test_dir, f_name)
            test_count += 1
        else:
            print("ERROR: Unrecognized split type:", split_type)
            return False

        if os.path.exists(o_file):
            skip_count += 1
        else:
            prepped = encode_midi(midi_path)
            o_stream = open(o_file, "wb")
            pickle.dump(prepped, o_stream)
            o_stream.close()

        if split_type == "train":
            for pitch in pitch_shift:
                for tempo in time_stretch:
                    train_count += 1

                    dir_split, extension_split = midi_path.rfind("/") + 1, midi_path.rfind(".midi") if ".midi" in midi_path else midi_path.rfind(".MIDI")
                    filepath, filename, extension = midi_path[:dir_split], midi_path[dir_split:extension_split], midi_path[extension_split:]
                    midi_aug_path = f"{filepath}Aug-{filename}-{pitch}-{tempo}{extension}"

                    # if (pitch == 0 and tempo == 1) and not os.path.exists(midi_aug_path):
                    #     augment_midi(midi_path, pitch, tempo)
                    #
                    # if pitch == 0 and tempo == 1:
                    #     continue

                    if pitch == 0 and tempo == 1:
                        continue
                    if not os.path.exists(midi_aug_path):
                        augment_midi(midi_path, pitch, tempo)
                        aug_count += 1

                    f_name_aug = midi_aug_path.split("/")[-1] + ".pickle"
                    o_file_aug = os.path.join(train_dir, f_name_aug)

                    if os.path.exists(o_file_aug):
                        skip_count += 1
                    else:
                        prepped_aug = encode_midi(midi_aug_path)
                        o_stream_aug = open(o_file_aug, "wb")
                        pickle.dump(prepped_aug, o_stream_aug)
                        o_stream_aug.close()

        total_count += 1
        if total_count % 50 == 0:
            print(total_count, "/", len(maestro_json))

    print("Num Train:", train_count)
    print("Num Val:", val_count)
    print("Num Test:", test_count)
    print("Num Augmented:", aug_count)
    print("Num Skipped:", skip_count)
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
