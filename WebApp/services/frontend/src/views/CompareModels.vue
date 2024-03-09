<template>
  <Header />
  <div class="home-container01">
    <span class="home-logo">Compare Models!</span>
  </div>
  <div class="center">
    <button class="button-stock" @click="updateChart">Load Charts</button>
    <div class="separator"></div>
    <!--<input class="text-input" v-model="selectedStock" placeholder="Please enter a stock">
    <input type="number" class="number-input" v-model.number="selectedDays" placeholder="Last n days">
    <button class="button-stock" @click="updateChart">Show Stock</button>!-->
    <span class="selection">Stock</span>
    <div class="selector">
      <select v-model="selectedStock" ref="selectorIn" @mouseover="changeCursor" @mouseleave="resetCursor">
        <option value="AAL" selected>
          American Airlines
        </option>
        <option value="AAPL">
          Apple
        </option>
        <option value="AMD">
          Advanced Micro Devices
        </option>
        <option value="C">
          Citigroup
        </option>
        <option value="MRNS">
          Marinus Pharmaceuticals Inc
        </option>
        <option value="NIO">
          Nio Inc
        </option>
        <option value="NVDA">
          Nvdia
        </option>
        <option value="SNAP">
          Snap Inc
        </option>
        <option value="SQ">
          Block Inc
        </option>
        <option value="TSLA">
          Tesla
        </option>
      </select>
    </div>
    <div class="separator"></div>
    <span class="selection">Time Interval</span>
    <div class="selector">
      <select v-model="selectedTime" ref="selectorIn" @mouseover="changeCursor" @mouseleave="resetCursor">
        <option value="D" selected>
          Daily
        </option>
        <option value="H">
          Hourly
        </option>
        <option value="M">
          Minutely
        </option>
      </select>
    </div>
    <div class="separator"></div>
    <button class="button-stock" @click="showAll">Check All</button>
    <button class="button-stock" @click="hideAll">Uncheck All</button>
    <!-- Add checkboxes for additional data points -->
    <div class="checkboxes">
      <div class="checkboxes1">
        <label>
          <input type="checkbox" v-model="showCNNLine"> CNN
        </label>
        <label>
          <input type="checkbox" v-model="showTransformerLine"> Transformer
        </label>
        <label>
          <input type="checkbox" v-model="showLSTMLine"> LSTM
        </label>
      </div>
      <div class="checkboxes4">
        <label>
          <input type="checkbox" v-model="showrandomForestLine"> Random Forest
        </label>
        <label>
          <input type="checkbox" v-model="showgradientBoostLine"> gradient Boost
        </label>
      </div>
    </div>
  </div>
  <!-- Combined Chart -->
  <div class="newChart">
    <DxChart v-if="showChart" :data-source="combinedData" title="Stock Price">
      <DxCommonSeriesSettings argument-field="DateTime" type="stock" />
      <DxSeries :name=selectedStock open-value-field="Open" high-value-field="High" low-value-field="Low"
        close-value-field="Close" argument-field="DateTime">
      </DxSeries>
      <DxSeries v-if="showCNNLine" :name="'CNN' + this.selectedStock" :data-source="combinedData" type="line"
        value-field="CNNValue" argument-field="date" :color="seriesColors[0]">
      </DxSeries>
      <DxSeries v-if="showTransformerLine" :name="'Transformer' + this.selectedStock" :data-source="combinedData"
        type="line" value-field="TransformerValue" argument-field="date" :color="seriesColors[1]">
      </DxSeries>
      <DxSeries v-if="showLSTMLine" :name="'LSTM' + this.selectedStock" :data-source="combinedData" type="line"
        value-field="LSTMValue" argument-field="date" :color="seriesColors[2]">
      </DxSeries>
      <DxSeries v-if="showrandomForestLine" :name="'randomForest' + this.selectedStock" :data-source="combinedData"
        type="line" value-field="randomForestValue" argument-field="date" :color="seriesColors[3]">
      </DxSeries>
      <DxSeries v-if="showgradientBoostLine" :name="'gradientBoost' + this.selectedStock" :data-source="combinedData"
        type="line" value-field="gradientBoostValue" argument-field="date" :color="seriesColors[4]">
      </DxSeries>
      <DxArgumentAxis :workdays-only="true">
        <DxTitle text="Time" />
        <DxLabel format="shortDate" />
      </DxArgumentAxis>
      <DxValueAxis name="price" position="left">
        <DxTitle text="US dollars" />
        <DxLabel>
          <DxFormat type="currency" />
        </DxLabel>
      </DxValueAxis>
      <DxTooltip :enabled="true" />
    </DxChart>
  </div>
  <div class="newChart">
    <DxChart v-if="showCNNLine && showChart" id="CNN-chart" :data-source="this.CNNData" :title="CNNchartTitle">
      <DxCommonSeriesSettings argument-field="date" type="line" />
      <DxSeries :name="'CNN Line'" value-field="value" argument-field="date" type="line" :color="seriesColors[0]">
      </DxSeries>
      <DxArgumentAxis :workdays-only="true">
        <DxTitle text="Time" />
        <DxLabel format="shortDate" />
      </DxArgumentAxis>
      <DxValueAxis name="price" position="left">
        <DxTitle text="US dollars" />
        <DxLabel>
          <DxFormat type="currency" />
        </DxLabel>
      </DxValueAxis>
      <DxTooltip :enabled="true" />
    </DxChart>
  </div>
  <div class="newChart">
    <DxChart v-if="showTransformerLine && showChart" id="Transformer-chart" :data-source="this.transformerData"
      :title="TransformerchartTitle">
      <DxCommonSeriesSettings argument-field="date" type="line" />
      <DxSeries :name="'Transformer Line'" value-field="value" argument-field="date" type="line"
        :color="seriesColors[1]">
      </DxSeries>
      <DxArgumentAxis :workdays-only="true">
        <DxTitle text="Time" />
        <DxLabel format="shortDate" />
      </DxArgumentAxis>
      <DxValueAxis name="price" position="left">
        <DxTitle text="US dollars" />
        <DxLabel>
          <DxFormat type="currency" />
        </DxLabel>
      </DxValueAxis>
      <DxTooltip :enabled="true" />
    </DxChart>
  </div>
  <div class="newChart">
    <DxChart v-if="showLSTMLine && showChart" id="LSTM-chart" :data-source="this.LSTMData" :title="LSTMchartTitle">
      <DxCommonSeriesSettings argument-field="date" type="line" />
      <DxSeries :name="'LSTM Line'" value-field="value" argument-field="date" type="line" :color="seriesColors[2]">
      </DxSeries>
      <DxArgumentAxis :workdays-only="true">
        <DxTitle text="Time" />
        <DxLabel format="shortDate" />
      </DxArgumentAxis>
      <DxValueAxis name="price" position="left">
        <DxTitle text="US dollars" />
        <DxLabel>
          <DxFormat type="currency" />
        </DxLabel>
      </DxValueAxis>
      <DxTooltip :enabled="true" />
    </DxChart>
  </div>
  <div class="newChart">
    <DxChart v-if="showrandomForestLine && showChart" id="Random Forest-chart" :data-source="this.randomForestData"
      :title="RandomForestchartTitle">
      <DxCommonSeriesSettings argument-field="date" type="line" />
      <DxSeries :name="'randomForest Line'" value-field="value" argument-field="date" type="line"
        :color="seriesColors[3]">
      </DxSeries>
      <DxArgumentAxis :workdays-only="true">
        <DxTitle text="Time" />
        <DxLabel format="shortDate" />
      </DxArgumentAxis>
      <DxValueAxis name="price" position="left">
        <DxTitle text="US dollars" />
        <DxLabel>
          <DxFormat type="currency" />
        </DxLabel>
      </DxValueAxis>
      <DxTooltip :enabled="true" />
    </DxChart>
  </div>
  <div class="newChart">
    <DxChart v-if="showgradientBoostLine && showChart" id="Gradient Boost-chart" :data-source="this.gradientBoostData"
      :title="GradientBoostchartTitle">
      <DxCommonSeriesSettings argument-field="date" type="line" />
      <DxSeries :name="'gradientBoost Line'" value-field="value" argument-field="date" type="line"
        :color="seriesColors[4]">
      </DxSeries>
      <DxArgumentAxis :workdays-only="true">
        <DxTitle text="Time" />
        <DxLabel format="shortDate" />
      </DxArgumentAxis>
      <DxValueAxis name="price" position="left">
        <DxTitle text="US dollars" />
        <DxLabel>
          <DxFormat type="currency" />
        </DxLabel>
      </DxValueAxis>
      <DxTooltip :enabled="true" />
    </DxChart>
  </div>
</template>
<script>

import DxChart, {
  DxCommonSeriesSettings,
  DxSeries,
  DxReduction,
  DxArgumentAxis,
  DxLabel,
  DxFormat,
  DxValueAxis,
  DxTitle,
  DxExport,
  DxTooltip,
} from 'devextreme-vue/chart';

import Header from './Header.vue'
import axios from "axios";
import Swal from "sweetalert2/dist/sweetalert2.js";
import { useMyPiniaStore } from "../store.js";

// Add the calculateMeanErrorAndMeanAbsoluteError function here
function calculateMeanErrorAndMeanAbsoluteError(actualValues, predictedValues) {
  if (actualValues.length !== predictedValues.length) {
    throw new Error("The lengths of actualValues and predictedValues should be the same.");
  }

  // Calculate errors for each data point
  const errors = actualValues.map((actual, index) => predictedValues[index] - actual);

  // Calculate mean error
  const meanError = errors.reduce((sum, error) => sum + error, 0) / errors.length;

  // Calculate mean absolute error
  const absoluteErrors = errors.map(error => Math.abs(error));
  const meanAbsoluteError = absoluteErrors.reduce((sum, error) => sum + error, 0) / absoluteErrors.length;

  return { meanError, meanAbsoluteError };
};

export default {
  components: {
    Header,
    DxChart,
    DxCommonSeriesSettings,
    DxSeries,
    DxReduction,
    DxArgumentAxis,
    DxLabel,
    DxFormat,
    DxValueAxis,
    DxTitle,
    DxExport,
    DxTooltip,
  },
  async created() {
    this.dataSource = [];
    this.selectedDays = null;
    this.transformerData = [];
    this.LSTMData = [];
    this.CNNData = [];
    this.randomForestData = [];
    this.gradientBoostData = [];
  },
  data() {
    return {
      dataSource: [],
      transformerData: [],
      LSTMData: [],
      CNNData: [],
      randomForestData: [],
      gradientBoostData: [],
      selectedDays: null,
      showChart: false,
      priceRange: { min: null, max: null },
      showCNNLine: false,
      CNNpredictionData: [], // Add this property to store prediction data
      showTransformerLine: false,
      TransformerpredictionData: [],
      showLSTMLine: false,
      LSTMpredictionData: [],
      showrandomForestLine: false,
      randomForestpredictionData: [],
      showgradientBoostLine: false,
      gradientBoostpredictionData: [],
      combinedData: [],
      store: useMyPiniaStore(),
      selectedTime: "H",
      selectedStock: "AAPL",
      seriesColors: ['#FF5733', '#33FF57', '#337AFF', '#FF33DC', '#33FFDC'], // Array of colors for each series
      meanError: null,
      meanAbsoluteError: null,
    };
  },
  computed: {
    CNNchartTitle() {
      return `CNN Chart; Mean Error: ${this.meanError}; Mean Absolute Error: ${this.meanAbsoluteError}`;
    },
    TransformerchartTitle() {
      return `Transformer Chart; Mean Error: ${this.meanError}; Mean Absolute Error: ${this.meanAbsoluteError}`;
    },
    LSTMchartTitle() {
      return `LSTM Chart; Mean Error: ${this.meanError}; Mean Absolute Error: ${this.meanAbsoluteError}`;
    },
    RandomForestchartTitle() {
      return `Random Forest Chart; Mean Error: ${this.meanError}; Mean Absolute Error: ${this.meanAbsoluteError}`;
    },
    GradientBoostchartTitle() {
      return `Gradient Boost Chart; Mean Error: ${this.meanError}; Mean Absolute Error: ${this.meanAbsoluteError}`;
    },
    tooltip: {
      enabled: true,
      // Customize tooltip appearance or behavior if needed
    },
  },

  mounted() {
    this.handleSelection();
  },

  methods: {
    // Method to calculate mean error and mean absolute error
    calculateErrorsAndDisplay(actualValues, predictedValues) {
      const { meanError, meanAbsoluteError } = calculateMeanErrorAndMeanAbsoluteError(actualValues, predictedValues);
      this.meanError = meanError.toFixed(2); // Round to 2 decimal places
      this.meanAbsoluteError = meanAbsoluteError.toFixed(2); // Round to 2 decimal places
    },

    // Example method to trigger calculation and display
    displayErrors() {
      // Example actual and predicted data
      const actualValues = [10, 20, 30, 40, 50];
      const predictedValues = [12, 18, 28, 38, 48];
      this.calculateErrorsAndDisplay(actualValues, predictedValues);
    },

    handleSelection() {
      console.log("SelectedTime:", this.selectedTime);
      console.log("SelectedStock:", this.selectedStock);
    },

    async updateCombinedData() {
      // Map dataSource points with all values set to null
      const combinedDataWithNull = this.dataSource.map(data => ({
        ...data,
        TransformerValue: null,
        CNNValue: null,
        LSTMValue: null,
        randomForestValue: null,
        gradientBoostValue: null,
      }));

      // Function to merge data points into combinedData
      const mergeDataPoints = (dataPoints, valueKey) => {
        for (let value in dataPoints) {
          dataPoints[value].forEach(Item => {
            const index = combinedDataWithNull.findIndex(data => data.date === Item.date);
            if (index !== -1) {
              combinedDataWithNull[index][valueKey] = Item.value;
            } else {
              combinedDataWithNull.push({
                date: Item.date,
                [valueKey]: Item.value,
              });
            }
          });
        }
      };

      // Merge all data points into combinedData
      mergeDataPoints(this.LSTMData, 'LSTMValue');
      mergeDataPoints(this.CNNData, 'CNNValue');
      mergeDataPoints(this.transformerData, 'TransformerValue');
      mergeDataPoints(this.randomForestData, 'randomForestValue');
      mergeDataPoints(this.gradientBoostData, 'gradientBoostValue');

      // Set the result to combinedData
      this.combinedData = combinedDataWithNull;

      console.log("combinedData");
      console.log(this.combinedData);
    },

    async hideAll() {
      this.showCNNLine = false;
      this.showTransformerLine = false;
      this.showLSTMLine = false;
      this.showrandomForestLine = false;
      this.showgradientBoostLine = false;
    },

    async showAll() {
      this.showTransformerLine = true;
      this.showrandomForestLine = true;
      this.showgradientBoostLine = true;
      this.showLSTMLine = true;
      this.showCNNLine = true;
    },

    async get_stock_data(stock_symbol, days_back) {
      try {
        const response = await axios.get(this.store.API + "/getStock", {
          params: {
            stock_symbol: this.selectedStock,
            start_date: "2021-02-04", // Replace with the start date of the prediction
            end_date: "2021-02-06", // Replace with the end date of the prediction
            resolution: this.selectedtime, // Replace with the resolution:
          }
        });
        console.log([response.data])
        return response.data
      } catch (error) {
        Swal.fire({
          title: "Error at getting data",
          text: error,
          icon: "info",
          showCloseButton: false,
          confirmButtonText: "Close",
          confirmButtonColor: "#d0342c",
        });
        if (
          error.response &&
          error.response.data &&
          error.response.data.detail
        ) {
          console.log(error.response.data.detail);
        } else {
          console.log(error);
        }
      }
    },

    async load_CNN_data() {
      try {
        const response = await axios.get(`${this.store.API}/predict/cnn`, {
          params: {
            stock_symbol: "[" + this.selectedStock + "]",
            start_date: "2021-02-04", // Replace with the start date of the prediction
            end_date: "2021-02-06", // Replace with the end date of the prediction
            resolution: this.selectedtime, // Replace with the resolution:
          }
        });

        console.log("Prediction cnn loaded");
        console.log(response.data);

        // Assuming the response.data is an object with date and close properties
        this.CNNData = Object.entries(response.data).map(([date, { value }]) => ({ date, value }));

        return response.data;
      } catch (error) {
        Swal.fire({
          title: "Error at predicting data",
          text: error,
          icon: "info",
          showCloseButton: false,
          confirmButtonText: "Close",
          confirmButtonColor: "#d0342c",
        });
        console.error(error);
      }
    },

    async load_LSTM_data() {
      try {
        const response = await axios.get(`${this.store.API}/predict/lstm`, {
          params: {
            stock_symbol: "[" + this.selectedStock + "]",
            start_date: "2021-02-04", // Replace with the start date of the prediction
            end_date: "2021-02-06", // Replace with the end date of the prediction
            resolution: this.selectedtime, // Replace with the resolution:
          }
        });

        console.log("Prediction lstm loaded");
        console.log(response.data);

        // Assuming the response.data is an object with date and close properties
        this.LSTMData = Object.entries(response.data).map(([date, { value }]) => ({ date, value }));

        return response.data;
      } catch (error) {
        Swal.fire({
          title: "Error at predicting data",
          text: error,
          icon: "info",
          showCloseButton: false,
          confirmButtonText: "Close",
          confirmButtonColor: "#d0342c",
        });
        console.error(error);
      }
    },

    async load_transformer_data() {
      try {
        const response = await axios.get(`${this.store.API}/predict/transformer`, {
          params: {
            stock_symbols: "[" + this.selectedStock + "]",
            start_date: "2021-02-04", // Replace with the start date of the prediction
            end_date: "2021-02-06", // Replace with the end date of the prediction
            resolution: this.selectedtime, // Replace with the resolution:
          }
        });

        console.log("Prediction transformer loaded");
        console.log(response.data);

        // Assuming the response.data is an object with date and close properties
        this.transformerData = Object.entries(response.data).map(([date, { value }]) => ({ date, value }));

        console.log("mapped transformerData:");
        console.log(this.transformerData);

        return response.data;
      } catch (error) {
        Swal.fire({
          title: "Error at predicting data",
          text: error,
          icon: "info",
          showCloseButton: false,
          confirmButtonText: "Close",
          confirmButtonColor: "#d0342c",
        });
        console.error(error);
      }
    },

    async load_randomForest_data() {
      try {
        const response = await axios.get(`${this.store.API}/predict/randomForest`, {
          params: {
            stock_symbol: "[" + this.selectedStock + "]",
            start_date: "2021-02-04", // Replace with the start date of the prediction
            end_date: "2021-02-06", // Replace with the end date of the prediction
            resolution: this.selectedtime, // Replace with the resolution:
          }
        });

        console.log("Prediction randomforst loaded");
        console.log(response.data);

        // Assuming the response.data is an object with date and close properties
        this.randomForestData = Object.entries(response.data).map(([date, { value }]) => ({ date, value }));

        console.log("mapped randomForestData:");
        console.log(this.randomForestData);

        return response.data;
      } catch (error) {
        Swal.fire({
          title: "Error at predicting data",
          text: error,
          icon: "info",
          showCloseButton: false,
          confirmButtonText: "Close",
          confirmButtonColor: "#d0342c",
        });
        console.error(error);
      }
    },

    async load_gradientBoost_data() {
      try {
        const response = await axios.get(`${this.store.API}/predict/gradientBoost`, {
          params: {
            stock_symbol: "[" + this.selectedStock + "]",
            start_date: "2021-02-04", // Replace with the start date of the prediction
            end_date: "2021-02-06", // Replace with the end date of the prediction
            resolution: this.selectedtime, // Replace with the resolution:
          }
        });

        console.log("Prediction gradientBoost loaded");
        console.log(response.data);

        // Assuming the response.data is an object with date and close properties
        this.gradientBoostData = Object.entries(response.data).map(([date, { value }]) => ({ date, value }));

        console.log("mapped gradientBoostData:");
        console.log(this.gradientBoostData);

        return response.data;
      } catch (error) {
        Swal.fire({
          title: "Error at predicting data",
          text: error,
          icon: "info",
          showCloseButton: false,
          confirmButtonText: "Close",
          confirmButtonColor: "#d0342c",
        });
        console.error(error);
      }
    },

    async load_data() {
      try {
        const response = await axios.get(this.store.API + "/load/data", {
          params: {
            stock_symbols: "[" + this.selectedStock + "]",
            start_date: "2021-02-04", // Replace with the start date of the prediction
            end_date: "2021-02-06", // Replace with the end date of the prediction
            resolution: this.selectedtime,
          }
        });
        console.log("### normal loaded ###")
        console.log([response.data])
        return response.data
      } catch (error) {
        Swal.fire({
          title: "Error at getting data",
          text: error,
          icon: "info",
          showCloseButton: false,
          confirmButtonText: "Close",
          confirmButtonColor: "#d0342c",
        });
        if (
          error.response &&
          error.response.data &&
          error.response.data.detail
        ) {
          console.log(error.response.data.detail);
        } else {
          console.log(error);
        }
      }
    },

    async updateChart() {
      console.log("##### SelectedTime: ", this.selectedTime);
      console.log("##### SelectedStock:", this.selectedStock);
      console.log("##### checkAllBoxes:", this.checkAllBoxes());

      if (this.selectedStock && this.selectedTime && this.checkAllBoxes()) {
        this.dataSource = await this.load_data();
        if (this.showCNNLine == true) {
          this.CNNData = await this.load_CNN_data();
        }
        if (this.showTransformerLine) {
          this.transformerData = await this.load_transformer_data();
        }
        if (this.showLSTMLine) {
          //this.LSTMData = await this.load_LSTM_data();
        }
        if (this.showrandomForestLine) {
          this.randomForestData = await this.load_randomForest_data();
        }
        if (this.showgradientBoostLine) {
          this.gradientBoostData = await this.load_gradientBoost_data();
        }

        console.log("datasource: " + this.dataSource)

        if (this.dataSource) {

          const prices = this.dataSource.flatMap(data => [data.open, data.close]);
          this.priceRange = {
            min: Math.min(...prices) * 0.5,
            max: Math.max(...prices) * 2
          };
          this.priceRangeKey = Math.random();
          console.log("price range: " + this.priceRange.min + " - " + this.priceRange.max)
        }

        await this.updateCombinedData();
        this.showChart = true;

      } else {
        Swal.fire({
          title: "Error at getting data",
          text: "Please select a stock, the time interval and a model",
          icon: "info",
          showCloseButton: false,
          confirmButtonText: "Close",
          confirmButtonColor: "#d0342c",
        });
      }
    },
    checkAllBoxes() {
      if (this.showCNNLine || this.showTransformerLine || this.showLSTMLine || this.showrandomForestLine || this.showgradientBoostLine) {
        return true;
      } else {
        return false;
      }
    },
  },
};
</script>

<style>
#chart {
  height: 30%;
}

.newChart {
  margin-top: 2%;
}

.checkboxes {
  margin-left: 1%;
  margin-right: 1%;
  padding: 0.5%;
  display: flex;
  justify-content: center;
}

.checkboxes label {
  display: flex;
  align-items: center;
}

.checkboxes input {
  margin-right: 5px;
  /* Adjust the margin as needed */
}

.center {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 10%;
  background-color: white;
  padding: 1%;
  border: 2px solid #ccc;
  /* Adjust border thickness and color as needed */
}

.selection {
  margin-right: 5px;
  margin-left: 5px;
}

.separator {
  height: 100px;
  width: 2px;
  /* Adjust the width of the separator */
  background-color: #817f7f;
  /* Adjust the color of the separator */
  margin: 0 10px;
  /* Adjust the margin around the separator */
}

select {
  color: #ffffff;
  background-color: grey;
  margin-right: 1%;
}

#select-id option {
  color: #ffffff;
}

input {
  color: #ffffff;
  margin-right: 1%;
}

.number-input {
  width: 10%;
  direction: rtl;
}

.home-container01 {
  width: 100%;
  flex: 0 0 auto;
  height: 90px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  font-size: 30px;
  /* Adjust the font size as needed */
  margin-top: 15px;
}


.text-input {
  color: #ffffff;
}

input::placeholder {
  color: #ffffff;
}

.number-input::-webkit-inner-spin-button,
.number-input::-webkit-outer-spin-button {
  -webkit-appearance: none;
  margin: 0;
}

.button-stock {
  background-color: grey;
  color: #ffffff;
  border: none;
  cursor: pointer;
  transition: background-color 0.3s ease;
  border-radius: 12px;
  box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.1);
  margin-right: 5px;
  margin-left: 5px;
}

.checkboxes4 {
  margin-right: 5px;
  margin-left: 5px;
}

.button-stock:hover {
  background-color: darkgrey;
}
</style>