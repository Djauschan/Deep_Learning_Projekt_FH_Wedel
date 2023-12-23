<template>
  <Header />
  <div class="center">
    <input class="text-input" v-model="selectedStock" placeholder="Please enter a stock">
    <input type="number" class="number-input" v-model.number="selectedDays" placeholder="Last n days">
    <button class="button-stock" @click="updateChart">Show Stock</button>
  </div>
  <DxChart v-if="showChart" id="chart" :data-source="dataSource" title="Stock Price">
    <DxCommonSeriesSettings argument-field="date" type="stock" />
    <DxSeries :name="selectedStock" open-value-field="open" high-value-field="high" low-value-field="low"
      close-value-field="close" axis="price" :customize-point="customizeStockPoint">
    </DxSeries>
    <DxSeries class="volume-chart" name="Volume" value-field="volume" argument-field="date" type="bar" axis="volume"
      :bar-width="5">
    </DxSeries>
    <DxArgumentAxis :workdays-only="true">
      <DxLabel format="shortDate" />
    </DxArgumentAxis>
    <DxValueAxis name="price" position="left" :min="priceRange.min" :max="priceRange.max" :key="priceRangeKey">
      <DxTitle text="US dollars" />
      <DxLabel>
        <DxFormat type="currency" />
      </DxLabel>
    </DxValueAxis>
    <DxValueAxis name="volume" position="right" :visualRange="volumeRange">
      <DxTitle text="Volume" />
    </DxValueAxis>
    <DxExport :enabled="true" />
    <DxTooltip :enabled="true" :customize-tooltip="customizeTooltip" location="edge" />
  </DxChart>
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
      volumeRange: { startValue: 0, endValue: null },
      priceRange: { min: null, max: null },
    };
  },
  methods: {
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
          const maxVolume = Math.max(...this.dataSource.map(data => data.volume));
          this.volumeRange.endValue = maxVolume * 5;

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