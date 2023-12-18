import numpy as np
def beam_search(model, input_seq, vocab, beam_width=3, max_length=50):
    """
    Beam search for neural machine translation.

    :param model: Trained RNN model for translation
    :param input_seq: Input sequence (English sentence)
    :param vocab: Vocabulary dictionary mapping words to indices
    :param beam_width: Number of sequences to keep at each step
    :param max_length: Maximum length of the output sequence
    :return: The best translation according to the beam search
    """
    # Start with an initial sequence with just the start token
    start_token = vocab['<start>']
    initial_seq = [[start_token, 0.0]]  # Each sequence is a list of tokens and its score

    # Beam search
    sequences = initial_seq
    for _ in range(max_length):
        all_candidates = []
        for seq in sequences:
            # Stop expanding this sequence if it's ended
            if seq[-1][0] == vocab['<end>']:
                all_candidates.append(seq)
                continue

            # Predict the next word and its probabilities
            # Note: model.predict() should be adapted according to how your model is implemented
            input_data = prepare_input(seq, vocab)  # Prepare the input data in the required format
            probabilities = model.predict(input_data)

            # Get top beam_width next words
            next_words = np.argsort(probabilities)[-beam_width:]

            # Create new sequences with these words
            for word in next_words:
                new_seq = seq + [(word, seq[-1][1] - np.log(probabilities[word]))]  # Subtract log prob
                all_candidates.append(new_seq)

        # Sort all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[-1][1])
        sequences = ordered[:beam_width]  # Keep top beam_width sequences

    # Choose the sequence with the highest score
    best_sequence = sequences[0]
    translated_sentence = [vocab_inv[idx] for idx, _ in best_sequence[1:-1]]  # Ignore start and end tokens
    return ' '.join(translated_sentence)
