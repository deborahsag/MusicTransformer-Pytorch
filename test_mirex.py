import unittest

from mirex import *

midi_root = 'dataset/maestro'
prompt_length = 512
continuation_length = 128
target_seq_length = prompt_length + continuation_length
random.seed(4242)

_, _, dataset = create_epiano_datasets(midi_root, target_seq_length, random_seq=True)


class TestMIREX(unittest.TestCase):

    def test_get_prompt(self):
        print("Running get_prompt test:\n")

        # Grab one random piece from dataset
        indexes = random.sample(range(len(dataset)), 1)
        pieces = [dataset[idx][0] for idx in indexes]
        piece = pieces[0]
        print(f"Type of pieces variable: {type(pieces)}")
        print(f"Type of piece variable: {type(piece)}\n")

        # Get prompt
        prompt = get_prompt(pieces, prompt_length)
        prompt_data = prompt.cpu().detach().numpy()
        prompt_confirm = piece[:prompt_length].cpu().detach().numpy()
        print(f"Type of prompt variable: {type(prompt)}")
        print(f"Type of prompt_data variable: {type(prompt_data)}\n")

        print("Assert prompt data")
        np.testing.assert_array_equal(prompt_data, prompt_confirm)
        print("Assert prompt length")
        self.assertEqual(len(prompt), prompt_length)

    def test_get_continuations(self):
        print("Running get_continuations test:\n")

        # Grab one random piece from dataset
        indexes = random.sample(range(len(dataset)), 4)
        pieces = [dataset[idx][0] for idx in indexes]
        piece = pieces[0]
        print(f"Type of pieces variable: {type(pieces)}")
        print(f"Type of piece variable: {type(piece)}")
        print(f"Shape of piece variable: {piece.shape}\n")

        # Grab its prompt and continuation
        prompt = get_prompt(pieces, prompt_length)
        continuations = get_continuations(pieces, prompt_length)
        continuation = continuations[0]
        print(f"Type of prompt variable: {type(prompt)}")
        print(f"Shape of prompt variable: {prompt.shape}\n")
        print(f"Type of continuations variable: {type(continuations)}")
        print(f"Length of continuations variable: {len(continuations)}")
        print(f"Type of continuation variable: {type(continuation)}")
        print(f"Shape of continuation variable: {continuation.shape}\n")

        # Test if prompt + continuation is equal to the piece
        whole_piece = torch.cat((prompt, continuation))
        print(f"Type of whole_piece variable: {type(whole_piece)}")
        print(f"Shape of whole_piece variable: {whole_piece.shape}\n")

        print("Assert that prompt + continuation is equal to the piece\n")
        torch.testing.assert_close(piece, whole_piece, rtol=0, atol=0)

        # print("Assert that prompt + other continuations are not equal to the piece\n")
        # false_piece_1 = torch.cat((prompt, continuations[1]))
        # false_piece_2 = torch.cat((prompt, continuations[2]))
        # false_piece_3 = torch.cat((prompt, continuations[3]))
        # torch.testing.assert_close(piece, false_piece_1, rtol=0, atol=0)
        # torch.testing.assert_close(piece, false_piece_2, rtol=0, atol=0)
        # torch.testing.assert_close(piece, false_piece_3, rtol=0, atol=0)

    # def test_compute_continuation_prob(self):
    #     res = 0
    #     assert res == 0


if __name__ == '__main__':
    unittest.main()
