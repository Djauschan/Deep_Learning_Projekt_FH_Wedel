import torch

def test_inference():
    # Laod model
    model = torch.load("../../data/output/models/Transformer_v6.pt")

    # get model dimensions
    encoder_dimensions = model.positional_encoding_encoder.pe.shape[1]
    encoder_seq_len = model.positional_encoding_encoder.pe.shape[0]
    decoder_dimensions = model.fc.out_features #model.positional_encoding_decoder.pe.shape[1]
    decoder_seq_len = model.positional_encoding_decoder.pe.shape[0]

    output_dimensions = model.fc.out_features
    output_seq_len = decoder_seq_len

    # generate random input
    ts = []
    ts.append(
        {
            'src': torch.ones((encoder_seq_len, encoder_dimensions)),
            'tgt': torch.ones((decoder_seq_len, decoder_dimensions))
        }
    )
    ts.append(
        {
            'src': torch.ones((encoder_seq_len, encoder_dimensions)) * 1000,
            'tgt': torch.ones((decoder_seq_len, decoder_dimensions)) * 1000
        }
    )
    ts.append(
        {
            'src': torch.ones((encoder_seq_len, encoder_dimensions)),
            'tgt': torch.ones((decoder_seq_len, decoder_dimensions))
        }
    )
    ts[2]['src'][0] = 1000
    ts[2]['tgt'][0] = 1000

    for t in ts:
        t['src'] = t['src'].unsqueeze(0)
        t['tgt'] = t['tgt'].unsqueeze(0)
        results = model.forward(**t)
        print(results)


if __name__ == "__main__":
    test_inference()