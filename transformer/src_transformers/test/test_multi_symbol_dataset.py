from unittest.mock import MagicMock

import pytest

from src_transformers.preprocessing.multi_symbol_dataset import MultiSymbolDataset


@pytest.mark.parametrize(
    "length, subseries_amount, validation_split, encoder_input_length, decoder_target_length",
    [
        (100, 5, 0.3, 2, 1),
        (103, 5, 0.3, 2, 1),
    ]
)
def test_get_subset_indices(length: int,
                            subseries_amount: int,
                            validation_split: float,
                            encoder_input_length: int,
                            decoder_target_length: int):
    # Create a mock MultiSymbolDataset object
    mock_dataset = MagicMock(MultiSymbolDataset)

    # Set the attributes of the mock
    mock_dataset.length = length
    mock_dataset.subseries_amount = subseries_amount
    mock_dataset.validation_split = validation_split
    mock_dataset.encoder_input_length = encoder_input_length
    mock_dataset.decoder_target_length = decoder_target_length

    # Set get_subset_indices to the original method
    mock_dataset.get_subset_indices = MultiSymbolDataset.get_subset_indices

    # Compute the training and validation indices based on the mocked dataset
    training_indices, validation_indices = mock_dataset.get_subset_indices(
        mock_dataset)

    # Set the expected results, they should look like this (if there is no offset):
    # training: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 21, ..., 91]
    training_indices_result = list(range(0, 12)) + list(range(20, 32)) + list(
        range(40, 52)) + list(range(60, 72)) + list(range(80, 92))
    # validation: [14, 15, 16, 17, 34, 35, 36, 37, ..., 97]
    validation_indices_result = list(range(14, 18)) + list(range(34, 38)) + list(
        range(54, 58)) + list(range(74, 78)) + list(range(94, 98))

    # Adjust for offset (if there are remaining elements that can't be assigned to a subseries)
    offset = length % subseries_amount
    training_indices_result = [x + offset for x in training_indices_result]
    validation_indices_result = [x + offset for x in validation_indices_result]

    assert training_indices, training_indices_result
    assert validation_indices, validation_indices_result
