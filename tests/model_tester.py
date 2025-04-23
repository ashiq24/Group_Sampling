from gsampling.models.model_handler import get_model
import unittest
import torch


class TestGcnnModel(unittest.TestCase):
    def test_model_creation(self):
        """
        Test the creation of the Gcnn model with default parameters.
        """
        num_channels = [32, 64]
        model = get_model(input_channel=3, num_channels=num_channels)
        # random input
        x = torch.randn(1, 3, 32, 32)
        output = model(x)
        self.assertEqual(output.shape, (1, 10))

        final_feature = model.get_feature(x)
        self.assertEqual(final_feature.shape[-1], num_channels[-1])

        dwn_group_types = [["dihedral", "cycle"], ["cycle", "cycle"]]
        model = get_model(
            input_channel=3, num_channels=num_channels, dwn_group_types=dwn_group_types
        )
        # random input
        x = torch.randn(1, 3, 32, 32)
        output = model(x)
        self.assertEqual(output.shape, (1, 10))

        final_feature = model.get_feature(x)
        self.assertEqual(final_feature.shape[-1], num_channels[-1])
