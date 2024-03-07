import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.metrics import mape
from darts.models import TransformerModel

from src_transformers.preprocessing.multi_symbol_dataset import MultiSymbolDataset

# dataset = MultiSymbolDataset.create_from_config(
#     read_all_files=False,
#     last_date=date(2023, 1, 3),
#     data_usage_ratio=0.5,
#     subseries_amount=5,
#     validation_split=0.2,
#     create_new_file=True,
#     data_file="data/output/darts_dataset.csv",
#     encoder_symbols=["AAPL", "AAL", "AMD", "C", "MRNA", "NIO", "NVDA", "SNAP", "SQ", "TSLA", "DXY", "SPX", "DJCIGC", "DJCISI", "DJCIEN", "DJCIIK", "DJI", "DJINET", "COMP", "W5000"],
#     decoder_symbols=["AAPL", "AAL", "AMD", "C", "MRNA", "NIO", "NVDA", "SNAP", "SQ", "TSLA"],
#     encoder_input_length=30,
#     decoder_target_length=10
# )

dataset = TimeSeries.from_csv("data/output/darts_dataset.csv",
                              time_col="timestamp",
                              freq="T",
                              fill_missing_dates=True,
                              fillna_value=0)

# dataset = dataset.tail(10000)

# print(dataset.columns)
# print(dataset.pd_dataframe().shape)

train, val = dataset.split_after(0.9)

# my_model = TransformerModel(
#     input_chunk_length=50,
#     output_chunk_length=5,
#     batch_size=1024,
#     n_epochs=10,
#     model_name="darts_transformer",
#     nr_epochs_val_period=10,
#     d_model=16,
#     nhead=8,
#     num_encoder_layers=2,
#     num_decoder_layers=2,
#     dim_feedforward=128,
#     dropout=0.1,
#     activation="relu",
#     random_state=42,
#     save_checkpoints=True,
#     log_tensorboard=True,
#     work_dir="runs"
# )

# my_model.fit(series=train, val_series=val, verbose=True)

# my_model.save("runs/darts_transformer/model.pt")

my_model = TransformerModel.load_from_checkpoint(model_name="darts_transformer", work_dir="runs", best=True)


def eval_model(model, series, val_series):
    pred_series = model.predict(n=len(val_series))
    plt.figure(figsize=(8, 5))

    pred_series = pred_series.pd_dataframe()
    pred_series_aapl = pred_series[["close AAPL", "close AMD"]]

    pred_series_aapl = TimeSeries.from_dataframe(pred_series_aapl)

    # series.plot(label="actual")
    pred_series.plot(label="forecast")

    # plt.title("MAPE: {:.2f}%".format(mape(pred_series, val_series)))

    plt.legend()
    plt.show()


eval_model(my_model, dataset, val)
