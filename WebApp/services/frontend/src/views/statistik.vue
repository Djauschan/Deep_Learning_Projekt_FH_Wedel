<template>
  <Header />
  <div class="center">
    <!--<input class="text-input" v-model="selectedStock" placeholder="Please enter a stock">
    <input type="number" class="number-input" v-model.number="selectedDays" placeholder="Last n days">
    <button class="button-stock" @click="updateChart">Show Stock</button>!-->
    <span class="selection">Stock</span>
    <div class="selector">
      <select v-model="selectedStock" ref="selectorIn" @mouseover="changeCursor" @mouseleave="resetCursor">
        <option value="Option 1" selected>
          Apple
        </option>
        <option value="Option 2">
          American Airlines
        </option>
        <option value="Option 3">
          Advanced Micro Devices
        </option>
        <option value="Option 4">
          Citigroup
        </option>
        <option value="Option 5">
          NVIDIA
        </option>
        <option value="Option 6">
          Snap
        </option>
        <option value="Option 7">
          Block
        </option>
        <option value="Option 8">
          Tesla
        </option>
      </select>
    </div>
    <span class="selection">Time Interval</span>
    <div class="selector">
      <select v-model="selectedTime" ref="selectorIn" @mouseover="changeCursor" @mouseleave="resetCursor">
        <option value="Option 1" selected>
          Daily
        </option>
        <option value="Option 2">
          Hourly
        </option>
        <option value="Option 3">
          Minutely
        </option>
      </select>
    </div>
    <button class="button-stock" @click="updateChart">Load Charts</button>
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
      <DxSeries :name="'aapl'" open-value-field="Open" high-value-field="High" low-value-field="Low"
        close-value-field="Close" argument-field="DateTime">
      </DxSeries>
      <DxSeries v-if="showCNNLine" :name="'CNN' + selectedStock" :data-source="combinedData" type="line"
        value-field="CNNValue" argument-field="date" :color="seriesColors[0]">
      </DxSeries>
      <DxSeries v-if="showTransformerLine" :name="'Transformer' + selectedStock" :data-source="combinedData"
        type="line" value-field="TransformerValue" argument-field="date" :color="seriesColors[1]">
      </DxSeries>
      <DxSeries v-if="showLSTMLine" :name="'LSTM' + selectedStock" :data-source="combinedData" type="line"
        value-field="LSTMValue" argument-field="date" :color="seriesColors[2]">
      </DxSeries>
      <DxSeries v-if="showrandomForestLine" :name="'randomForest' + selectedStock"
        :data-source="combinedData" type="line" value-field="randomForestValue" argument-field="date" :color="seriesColors[3]">
      </DxSeries>
      <DxSeries v-if="showgradientBoostLine" :name="'gradientBoost' + selectedStock"
        :data-source="combinedData" type="line" value-field="gradientBoostValue" argument-field="date" :color="seriesColors[4]">
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
    <DxChart v-if="showCNNLine && showChart" id="CNN-chart" :data-source="this.CNNData" title="CNN Chart">
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
   <DxChart v-if="showTransformerLine && showChart" id="Transformer-chart" :data-source="this.transformerData" title="Transformer Chart">
    <DxCommonSeriesSettings argument-field="date" type="line" />
    <DxSeries :name="'Transformer Line'" value-field="value" argument-field="date" type="line" :color="seriesColors[1]">
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
    <DxChart v-if="showLSTMLine && showChart" id="LSTM-chart" :data-source="this.LSTMData" title="LSTM Chart">
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
    <DxChart v-if="showrandomForestLine && showChart" id="Random Forest-chart"
      :data-source="this.randomForestData" title="Random Forest Chart">
      <DxCommonSeriesSettings argument-field="date" type="line" />
      <DxSeries :name="'randomForest Line'" value-field="value" argument-field="date" type="line" :color="seriesColors[3]">
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
    <DxChart v-if="showgradientBoostLine && showChart" id="Gradient Boost-chart"
      :data-source="this.gradientBoostData" title="Gradient Boost Chart">
      <DxCommonSeriesSettings argument-field="date" type="line" />
      <DxSeries :name="'gradientBoost Line'" value-field="value" argument-field="date" type="line" :color="seriesColors[4]">
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

//import { dataSource } from './data.js';
import Header from './Header.vue'
import axios from "axios";
import Swal from "sweetalert2/dist/sweetalert2.js";
import { useMyPiniaStore } from "../store.js";

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
    this.transformerData =[];
    this.LSTMData =[];
    this.CNNData =[];
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
      selectedStock: "",
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
      selectedTime: 'Option 1',
      selectedStock: 'Option 1',
      seriesColors: ['#FF5733', '#33FF57', '#337AFF', '#FF33DC', '#33FFDC'] // Array of colors for each series
    };
  },
  computed: {
    tooltip: {
          enabled: true,
          // Customize tooltip appearance or behavior if needed
        },
  },  
  
  mounted(){
    this.handleSelection();
  },

  methods: {

    handleSelection(){
      console.log("SelectedTime:", this.selectedTime);
      console.log("SelectedStock:", this.SelectedStock);
    },

    async updateCombinedData() {
    // Map dataSource points with TransformerValue set to null
    const combinedDataWithNull = this.dataSource.map(data => ({
      ...data,
      TransformerValue: null,
      CNNValue: null,
      LSTMValue: null,
      randomForestValue: null,
      gradientBoostValue: null,
    }));

    // Merge transformerData points into combinedData
    this.transformerData.forEach(transformerDataPoint => {
      const index = combinedDataWithNull.findIndex(data => data.date === transformerDataPoint.date);
      if (index !== -1) {
        // If date exists in combinedData, update TransformerValue
        combinedDataWithNull[index].TransformerValue = transformerDataPoint.value;
      } else {
        // If date doesn't exist in combinedData, add a new point
        combinedDataWithNull.push({
          date: transformerDataPoint.date,
          TransformerValue: transformerDataPoint.value,
        });
      }
    });

    this.LSTMData.forEach(LSTMDataPoint => {
      const index = combinedDataWithNull.findIndex(data => data.date === LSTMDataPoint.date);
      if (index !== -1) {
        // If date exists in combinedData, update TransformerValue
        combinedDataWithNull[index].LSTMValue = LSTMDataPoint.value;
      } else {
        // If date doesn't exist in combinedData, add a new point
        combinedDataWithNull.push({
          date: LSTMDataPoint.date,
          LSTMValue: LSTMDataPoint.value,
        });
      }
    });

    this.CNNData.forEach(CNNDataPoint => {
      const index = combinedDataWithNull.findIndex(data => data.date === CNNDataPoint.date);
      if (index !== -1) {
        // If date exists in combinedData, update TransformerValue
        combinedDataWithNull[index].CNNValue = CNNDataPoint.value;
      } else {
        // If date doesn't exist in combinedData, add a new point
        combinedDataWithNull.push({
          date: CNNDataPoint.date,
          CNNValue: CNNDataPoint.value,
        });
      }
    });

    this.randomForestData.forEach(randomForestDataPoint => {
      const index = combinedDataWithNull.findIndex(data => data.date === randomForestDataPoint.date);
      if (index !== -1) {
        // If date exists in combinedData, update TransformerValue
        combinedDataWithNull[index].randomForestValue = randomForestDataPoint.value;
      } else {
        // If date doesn't exist in combinedData, add a new point
        combinedDataWithNull.push({
          date: randomForestDataPoint.date,
          randomForestValue: randomForestDataPoint.value,
        });
      }
    });

    this.gradientBoostData.forEach(gradientBoostDataPoint => {
      const index = combinedDataWithNull.findIndex(data => data.date === gradientBoostDataPoint.date);
      if (index !== -1) {
        // If date exists in combinedData, update TransformerValue
        combinedDataWithNull[index].gradientBoostValue = gradientBoostDataPoint.value;
      } else {
        // If date doesn't exist in combinedData, add a new point
        combinedDataWithNull.push({
          date: gradientBoostDataPoint.date,
          gradientBoostValue: gradientBoostDataPoint.value,
        });
      }
    });

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
            stock_symbol: stock_symbol,
            days_back: days_back,
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
            stock_symbol: "AAPL"
          }
        });

        console.log("Prediction loaded");
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
            stock_symbol: "AAPL"
          }
        });

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
            stock_symbol: "AAPL"
          }
        });

        console.log("Prediction loaded");
        console.log(response.data);

        // Assuming the response.data is an object with date and close properties
        this.transformerData = Object.entries(response.data).map(([date, { value }]) => ({ date, value }));

        console.log("mapped");
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
            stock_symbol: "AAPL"
          }
        });

        // Assuming the response.data is an object with date and close properties
        this.randomForestData = Object.entries(response.data).map(([date, { value }]) => ({ date, value }));

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
            stock_symbol: "AAPL"
          }
        });

        // Assuming the response.data is an object with date and close properties
        this.gradientBoostData = Object.entries(response.data).map(([date, { value }]) => ({ date, value }));

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
        });
        console.log("normal loaded")
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
      this.dataSource = await this.load_data();
      this.transformerData = await this.load_transformer_data();
      this.randomForestData = await this.load_randomForest_data();
      this.gradientBoostData = await this.load_gradientBoost_data();
      this.LSTMData = await this.load_LSTM_data();
      this.CNNData = await this.load_CNN_data();
      

      console.log(this.dataSource)
      
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
      //}
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
  background-color: lightgray;
  padding: 1%;
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
  background-color: grey;
  margin-right: 1%;
}

.number-input {
  width: 10%;
  direction: rtl;
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
}

.button-stock:hover {
  background-color: darkgrey;
}
</style>