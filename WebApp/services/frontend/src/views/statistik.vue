<template>
  <Header />
  <div class="center">
    <input class="text-input" v-model="selectedStock" placeholder="Please enter a stock">
    <input type="number" class="number-input" v-model.number="selectedDays" placeholder="Last n days">
    <button class="button-stock" @click="updateChart">Show Stock</button>
    <button class="button-stock" @click="useTestData">Use Test Data</button>
    <button class="button-stock" @click="showAll">Show All</button>
    <button class="button-stock" @click="hideAll">Hide All</button>
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
      <div class="checkboxes2">
    <label>
      <input type="checkbox" v-model="showArimaLine"> Arima
    </label>
    <label>
      <input type="checkbox" v-model="showStatistischeMethodenLine"> Statistische Methoden
    </label>
    <label>
      <input type="checkbox" v-model="showRandomLine"> Random Line
    </label>
  </div>
    </div>
  </div>
   <!-- Combined Chart -->
   <div class="newChart">
      <DxChart v-if="showChart" :data-source="combinedData" title="Stock Price">
        <DxCommonSeriesSettings argument-field="date" type="stock" />
        <DxSeries :name="selectedStock" open-value-field="open" high-value-field="high" low-value-field="low"
          close-value-field="close" argument-field="date">
        </DxSeries>
        <DxSeries v-if="showCNNLine" :name="'CNN Line - ' + selectedStock" :data-source="combinedData" type="line"
          value-field="CNNValue" argument-field="date">
        </DxSeries>
        <DxSeries v-if="showTransformerLine" :name="'Transformer Line - ' + selectedStock" :data-source="combinedData" type="line"
          value-field="TransformerValue" argument-field="date">
        </DxSeries>
        <DxSeries v-if="showLSTMLine" :name="'LSTM Line - ' + selectedStock" :data-source="combinedData" type="line"
          value-field="LSTMValue" argument-field="date">
        </DxSeries>
        <DxSeries v-if="showArimaLine" :name="'Arima Line - ' + selectedStock" :data-source="combinedData" type="line"
          value-field="ArimaValue" argument-field="date">
        </DxSeries>
        <DxSeries v-if="showStatistischeMethodenLine" :name="'Statistische Methoden Line - ' + selectedStock" :data-source="combinedData" type="line"
          value-field="StatistischeMethodenValue" argument-field="date">
        </DxSeries>
        <DxSeries v-if="showRandomLine" :name="'Random Line - ' + selectedStock" :data-source="combinedData" type="line"
          value-field="randomValue" argument-field="date">
        </DxSeries>
        <DxArgumentAxis :workdays-only="true">
          <DxLabel format="shortDate" />
        </DxArgumentAxis>
        <DxValueAxis name="price" position="left">
          <DxTitle text="US dollars" />
          <DxLabel>
            <DxFormat type="currency" />
          </DxLabel>
        </DxValueAxis>
      </DxChart>
  </div>
  <div class="newChart">
  <DxChart v-if="showCNNLine && showChart" id="CNN-chart" :data-source="lineChartDataSource" title="CNN Chart">
    <DxCommonSeriesSettings argument-field="date" type="line" />
    <DxSeries :name="'CNN Line'" value-field="lineChartDataField" argument-field="date" type="line">
    </DxSeries>
  </DxChart>
  </div>
  <div class="newChart">
  <DxChart v-if="showTransformerLine && showChart" id="Transformer-chart" :data-source="lineChartDataSource" title="Transformer Chart">
    <DxCommonSeriesSettings argument-field="date" type="line" />
    <DxSeries :name="'Transformer Line'" value-field="lineChartDataField" argument-field="date" type="line">
    </DxSeries>
  </DxChart>
  </div>
  <div class="newChart">
  <DxChart v-if="showLSTMLine && showChart" id="LSTM-chart" :data-source="lineChartDataSource" title="LSTM Chart">
    <DxCommonSeriesSettings argument-field="date" type="line" />
    <DxSeries :name="'LSTM Line'" value-field="lineChartDataField" argument-field="date" type="line">
    </DxSeries>
  </DxChart>
  </div>
  <div class="newChart">
  <DxChart v-if="showArimaLine && showChart" id="Arima-chart" :data-source="lineChartDataSource" title="Arima Chart">
    <DxCommonSeriesSettings argument-field="date" type="line" />
    <DxSeries :name="'Arima Line'" value-field="lineChartDataField" argument-field="date" type="line">
    </DxSeries>
  </DxChart>
  </div>
  <div class="newChart">
  <DxChart v-if="showStatistischeMethodenLine && showChart" id="Statistische Methoden-chart" :data-source="lineChartDataSource" title="Statistische Methoden Chart">
    <DxCommonSeriesSettings argument-field="date" type="line" />
    <DxSeries :name="'StatistischeMethoden Line'" value-field="lineChartDataField" argument-field="date" type="line">
    </DxSeries>
  </DxChart>
  </div>
  <div class="newChart">
  <DxChart v-if="showRandomLine && showChart" id="Random-chart" :data-source="lineChartDataSource" title="Random Chart">
    <DxCommonSeriesSettings argument-field="date" type="line" />
    <DxSeries :name="'Line Chart'" value-field="lineChartDataField" argument-field="date" type="line">
    </DxSeries>
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

import { dataSource } from './data.js';
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
    this.dataSource = null;
    this.selectedDays = null;
  },
  data() {
    return {
      dataSource,
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
      showArimaLine: false,
      ArimapredictionData: [],
      showStatistischeMethodenLine: false,
      StatistischeMethodenpredictionData: [],
      showRandomLine: false,
      randomLineData: [],
    };
  },
  computed: {
    combinedData() {
      const combinedData = this.dataSource.map(data => {
        const randomValue = this.showRandomLine ? Math.random() * 80 + 70: 0;
        return {
          date: data.date,
          open: data.open,
          high: data.high,
          low: data.low,
          close: data.close,
          volume: data.volume,
          randomValue: randomValue,
        };
      });
      return combinedData;
    },
  },
  methods: {
    async hideAll(){
      this.showCNNLine = false;
      this.showTransformerLine = false;
      this.showLSTMLine = false;
      this.showArimaLine = false;
      this.showStatistischeMethodenLine = false;
      this.showRandomLine = false;
    },

    async showAll(){
      this.showCNNLine = true;
      this.showTransformerLine = true;
      this.showLSTMLine = true;
      this.showArimaLine = true;
      this.showStatistischeMethodenLine = true;
      this.showRandomLine = true;
    },

    async useTestData() {
    // Replace this with your actual test data for stock prices
const testData = [
  { date: '2022-01-01', open: 100, close: 110, high: 120, low: 90, volume: 1000000 },
  { date: '2022-01-02', open: 110, close: 105, high: 115, low: 100, volume: 1200000 },
  { date: '2022-01-03', open: 105, close: 112, high: 115, low: 100, volume: 900000 },
  { date: '2022-01-04', open: 112, close: 115, high: 120, low: 105, volume: 1100000 },
  // Add more test data as needed
];

// Set the data source with the test data
this.dataSource = testData;

// Dummy data for CNN prediction
const CNNpredictionTestData = [
  { date: '2022-01-01', predictedValue: 105 },
  { date: '2022-01-02', predictedValue: 108 },
  { date: '2022-01-03', predictedValue: 110 },
  { date: '2022-01-04', predictedValue: 113 },
  // Add more dummy data as needed
];
this.CNNpredictionData = CNNpredictionTestData;

// Dummy data for Transformer prediction
const TransformerpredictionTestData = [
  { date: '2022-01-01', predictedValue: 102 },
  { date: '2022-01-02', predictedValue: 107 },
  { date: '2022-01-03', predictedValue: 105 },
  { date: '2022-01-04', predictedValue: 110 },
  // Add more dummy data as needed
];
this.TransformerpredictionData = TransformerpredictionTestData;

// Customize other properties as needed
this.showChart = true;
const maxVolume = Math.max(...this.dataSource.map(data => data.volume));

const prices = this.dataSource.flatMap(data => [data.open, data.close]);
this.priceRange = {
  min: Math.min(...prices) * 0.5,
  max: Math.max(...prices) * 2
};
this.priceRangeKey = Math.random()
  },

    customizeTooltip(pointInfo) {
      if (!pointInfo.valueText.includes('h:')) {
        return {
          text: `Volume: ${(pointInfo.valueText / 1000000).toFixed(2)}M`
        };
      } else {
        return {
          text:
            `
            Date: $${(pointInfo.argument)}<br/>
            Open: $${Number(pointInfo.openValue).toFixed(2)}<br/>
            Close: $${Number(pointInfo.closeValue).toFixed(2)}<br/>
            High: $${Number(pointInfo.highValue).toFixed(2)}<br/>
            Low: $${Number(pointInfo.lowValue).toFixed(2)}<br/>
            `,
        };
      }
    },
    customizeStockPoint(point) {
      const stockData = this.dataSource.find(data => data.date === point.argument);
      if (stockData) {
        const color = stockData.close >= stockData.open ? 'green' : 'red';
        return {
          color: color,
          hoverStyle: {
            color: color
          }
        };
      }
    },
    async get_stock_data(stock_symbol, days_back) {
      const store = useMyPiniaStore();
      try {
        const response = await axios.get(store.API + "/getStock", {
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
    async updateChart() {
      if (this.selectedStock == "" || !this.selectedDays) {
        Swal.fire({
          title: "Error",
          text: "Please fill out all fields",
          icon: "info",
          showCloseButton: false,
          confirmButtonText: "Close",
          confirmButtonColor: "#d0342c",
        });
      } else {
        console.log(this.selectedStock, this.selectedDays)
        this.dataSource = await this.get_stock_data(this.selectedStock, this.selectedDays);

        if (this.dataSource) {

          const prices = this.dataSource.flatMap(data => [data.open, data.close]);
          this.priceRange = {
            min: Math.min(...prices) * 0.5,
            max: Math.max(...prices) * 2
          };
          this.priceRangeKey = Math.random();
          console.log(this.priceRange)
        }

        this.showChart = true;
      }
    },
  },
};
</script>
<style>
#chart {
  height: 30%;
}

.newChart{
  margin-top: 2%;
}

.checkboxes{
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
    margin-right: 5px; /* Adjust the margin as needed */
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