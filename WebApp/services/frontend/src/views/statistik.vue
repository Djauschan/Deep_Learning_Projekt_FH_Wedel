<template>
  <Header />
  <DxChart id="chart" :data-source="get_stock_data()" title="Stock Price">
    <DxCommonSeriesSettings argument-field="date" type="stock" />
    <DxSeries name="E-Mart" open-value-field="open" high-value-field="high" low-value-field="low" close-value-field="close">
      <DxReduction color="red" />
    </DxSeries>
    <DxArgumentAxis :workdays-only="true">
      <DxLabel format="shortDate" />
    </DxArgumentAxis>
    <DxValueAxis :tick-interval="1">
      <DxTitle text="US dollars" />
      <DxLabel>
        <DxFormat :precision="0" type="currency" />
      </DxLabel>
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
import { provide, inject } from 'vue'

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
  data() {
    return {
      dataSource,
    };
  },
  methods: {
    customizeTooltip(pointInfo) {
      return {
        text: `Open: $${pointInfo.openValue}<br/>
Close: $${pointInfo.closeValue}<br/>
High: $${pointInfo.highValue}<br/>
Low: $${pointInfo.lowValue}<br/>`,
      };
    },
    async get_stock_data() {
      const store = useMyPiniaStore();
      try {
        const response = await axios.get(store.API + "/getStock", {
          params: {
            stock_symbol: 'aapl',
            days_back: '10',
          }
        });
        console.log("localStorage.API: " + store.API)
        console.log(response.data)
        return response.data.date
      } catch (error) {
        Swal.fire({
          title: "Error at getting data",
          text: error,
          icon: "info",
          iconColor: "#d0342c",
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
  },
};
</script>
<style>
#chart {
  height: 440px;
}
</style>