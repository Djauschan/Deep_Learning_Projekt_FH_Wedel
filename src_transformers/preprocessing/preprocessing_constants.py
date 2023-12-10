from typing import Final
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer

SCALER_OPTIONS: Final[list] = [MinMaxScaler,
                               StandardScaler,
                               QuantileTransformer,
                               PowerTransformer]