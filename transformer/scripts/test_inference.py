import torch


def test_inference():
    # Laod model
    model = torch.load("../data/output/models/Transformer_v17.pt")

    # get model dimensions
    encoder_dimensions = model.positional_encoding_encoder.pe.shape[1]
    encoder_seq_len = model.positional_encoding_encoder.pe.shape[0]
    # model.positional_encoding_decoder.pe.shape[1]
    decoder_dimensions = model.fc.out_features
    decoder_seq_len = model.positional_encoding_decoder.pe.shape[0]

    output_dimensions = model.fc.out_features
    output_seq_len = decoder_seq_len

    torch.manual_seed(69)
    src_tensor = torch.rand((encoder_seq_len, encoder_dimensions))
    torch.manual_seed(68)
    tgt_tensor = torch.rand((decoder_seq_len, decoder_dimensions))

    # generate random input
    ts = []
    ts.append(
        {
            'src': src_tensor.clone(),
            'tgt': tgt_tensor.clone()
        }
    )
    ts.append(
        {
            'src': torch.mul(src_tensor.clone(), 1000),
            'tgt': torch.mul(tgt_tensor.clone(), 1000)
        }
    )
    ts.append(
        {
            'src': src_tensor.clone(),
            'tgt': tgt_tensor.clone()
        }
    )
    ts[2]['src'][0] = torch.mul(ts[2]['src'][0], 1000)
    ts[2]['tgt'][0] = torch.mul(ts[2]['tgt'][0], 1000)

    for t in ts:
        t['src'] = t['src'].unsqueeze(0)
        t['tgt'] = t['tgt'].unsqueeze(0)
        results = model.forward(**t)
        print(results)


if __name__ == "__main__":
    test_inference()
