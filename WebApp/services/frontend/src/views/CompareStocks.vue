<template>
  <Header />
  <div class="center">
    <!--<input class="text-input" v-model="selectedStock" placeholder="Please enter a stock">
    <input type="number" class="number-input" v-model.number="selectedDays" placeholder="Last n days">
    <button class="button-stock" @click="updateChart">Show Stock</button>!-->
    <!-- Add checkboxes for additional data points -->
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
    <br>
    <span class="selection">Model</span>
    <div class="selector">
      <select v-model="selectedModel" ref="selectorIn" @mouseover="changeCursor" @mouseleave="resetCursor">
        <option value="Option 1" selected>
          Transformer
        </option>
        <option value="Option 2">
          CNN
        </option>
        <option value="Option 3">
          Random Forest
        </option>
        <option value="Option 4">
          Gradient Boost
        </option>
      </select>
    </div>
    <div class="checkboxes">
      <div class="checkboxes1">
        <label>
          <input type="checkbox" v-model="showAAPLLine"> Apple
        </label>
        <label>
          <input type="checkbox" v-model="showAALLine"> American Airlines
        </label>
        <label>
          <input type="checkbox" v-model="showAMDLine"> Advanced Micro Devices
        </label>
      </div>
      <div class="checkboxes2">
        <label>
          <input type="checkbox" v-model="showCLine"> Citigroup
        </label>
        <label>
          <input type="checkbox" v-model="showNVDALine"> NVIDIA
        </label>
        <label>
          <input type="checkbox" v-model="showSnapLine"> Snap
        </label>
      </div>
      <div class="checkboxes3">
        <label>
          <input type="checkbox" v-model="showSQLine"> Block
        </label>
        <label>
          <input type="checkbox" v-model="showTSLALine"> Tesla
        </label>
      </div>
      <button class="button-stock" @click="showAll">Check All</button>
      <br>
      <button class="button-stock" @click="hideAll">Uncheck All</button>
      <br>
      <button class="button-stock" @click="updateChart">Show Predictions</button>
    </div>
  </div>
  <!-- Combined Chart -->
  <div class="newChart">
    <DxChart v-if="showChart" :data-source="combinedData" title="Stock Price">
      <DxCommonSeriesSettings argument-field="DateTime" type="stock" />
       <!--real werte anzeigen?
      <DxSeries :name="'aapl'" open-value-field="Open" high-value-field="High" low-value-field="Low"
        close-value-field="Close" argument-field="DateTime">
      </DxSeries> -->

      <DxSeries v-if="showAAPLLine" :name="'AAPL Line' + selectedStock" :data-source="combinedData" type="line"
        value-field="AAPLValue" argument-field="date">
      </DxSeries>
      <DxSeries v-if="showAALLine" :name="'AAL Line' + selectedStock" :data-source="combinedData"
        type="line" value-field="AALValue" argument-field="date">
      </DxSeries>
      <DxSeries v-if="showAMDLine" :name="'AMD Line' + selectedStock" :data-source="combinedData" type="line"
        value-field="AMDValue" argument-field="date">
      </DxSeries>
      <DxSeries v-if="showCLine" :name="'C Line' + selectedStock" :data-source="combinedData" type="line"
        value-field="CValue" argument-field="date">
      </DxSeries>
      <DxSeries v-if="showNVDALine" :name="'NVDA Line' + selectedStock"
        :data-source="combinedData" type="line" value-field="NVDAValue" argument-field="date">
      </DxSeries>
      <DxSeries v-if="showSnapLine" :name="'Snap Line' + selectedStock"
        :data-source="combinedData" type="line" value-field="SnapValue" argument-field="date">
      </DxSeries>
      <DxSeries v-if="showSQLine" :name="'SQ Line' + selectedStock"
        :data-source="combinedData" type="line" value-field="SQValue" argument-field="date">
      </DxSeries>
      <DxSeries v-if="showTSLALine" :name="'TSLA Line' + selectedStock"
        :data-source="combinedData" type="line" value-field="TSLAValue" argument-field="date">
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
    <DxChart v-if="showAAPLLine && showChart" id="AAPL-chart" :data-source="this.AAPLData" title="AAPL Chart">
      <DxCommonSeriesSettings argument-field="date" type="line" />
      <DxSeries :name="'AAPL Line'" value-field="value" argument-field="date" type="line">
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
   <DxChart v-if="showAALLine && showChart" id="AAL-chart" :data-source="this.AALData" title="AAL Chart">
    <DxCommonSeriesSettings argument-field="date" type="line" />
    <DxSeries :name="'AAL Line'" value-field="value" argument-field="date" type="line">
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
    <DxChart v-if="showAMDLine && showChart" id="AMD-chart" :data-source="this.AMDData" title="AMD Chart">
      <DxCommonSeriesSettings argument-field="date" type="line" />
      <DxSeries :name="'AMD Line'" value-field="value" argument-field="date" type="line">
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
    <DxChart v-if="showCLine && showChart" id="C-chart" :data-source="this.CData" title="C Chart">
      <DxCommonSeriesSettings argument-field="date" type="line" />
      <DxSeries :name="'C Line'" value-field="value" argument-field="date" type="line">
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
    <DxChart v-if="showNVDALine && showChart" id="Historic Average-chart"
      :data-source="this.NVDAData" title="Historic Average Chart">
      <DxCommonSeriesSettings argument-field="date" type="line" />
      <DxSeries :name="'NVDA Line'" value-field="value" argument-field="date" type="line">
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
    <DxChart v-if="showSnapLine && showChart" id="Window Average-chart"
      :data-source="this.SnapData" title="Window Average Chart">
      <DxCommonSeriesSettings argument-field="date" type="line" />
      <DxSeries :name="'Snap Line'" value-field="value" argument-field="date" type="line">
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
    <DxChart v-if="showSQLine && showChart" id="SQ-chart"
      :data-source="this.SQData" title="SQ Chart">
      <DxCommonSeriesSettings argument-field="date" type="line" />
      <DxSeries :name="'SQ Line'" value-field="value" argument-field="date" type="line">
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
    <DxChart v-if="showTSLALine && showChart" id="TSLA-chart"
      :data-source="this.TSLAData" title="TSLA Chart">
      <DxCommonSeriesSettings argument-field="date" type="line" />
      <DxSeries :name="'TSLA Line'" value-field="value" argument-field="date" type="line">
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
    this.AALData =[];
    this.NVDAData = [];
    this.svmData = [];
    this.SnapData = [];
    this.SQData = [];
    this.TSLAData = [];
    this.CData = [];
    this.AMDData =[];
    this.AAPLData =[];
  },
  data() {
    return {
      dataSource: [],
      AALData: [],
      AMDData: [],
      AAPLData: [],
      NVDAData: [],
      SnapData: [],
      SQData: [],
      TSLAData: [],
      CData: [],
      selectedStock: "",
      selectedDays: null,
      showChart: false,
      priceRange: { min: null, max: null },
      showAAPLLine: false,
      AAPLpredictionData: [],
      showAALLine: false,
      AALpredictionData: [],
      showAMDLine: false,
      AMDpredictionData: [],
      showCLine: false,
      CpredictionData: [],
      showNVDALine: false,
      NVDApredictionData: [],
      showSnapLine: false,
      SnappredictionData: [],
      showSQLine: false,
      SQpredictionData: [],
      showTSLALine: false,
      TSLApredictionData: [],
      combinedData: [],
      store: useMyPiniaStore(),
      selectedTime: 'Option 1',
      selectedModel: 'Option 1',
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
      console.log("SelectedModel:", this.selectedModel);
    },

    async updateCombinedData() {
    // Map dataSource points with AALValue set to null
    const combinedDataWithNull = this.dataSource.map(data => ({
      ...data,
      AALValue: null,
      AAPLValue: null,
      AMDValue: null,
      CValue: null,
      NVDAValue: null,
      SnapValue: null,
      SQValue: null,
      TSLAValue: null,
    }));

    // Merge AALData points into combinedData
    this.AALData.forEach(AALDataPoint => {
      const index = combinedDataWithNull.findIndex(data => data.date === AALDataPoint.date);
      if (index !== -1) {
        // If date exists in combinedData, update AALValue
        combinedDataWithNull[index].AALValue = AALDataPoint.value;
      } else {
        // If date doesn't exist in combinedData, add a new point
        combinedDataWithNull.push({
          date: AALDataPoint.date,
          AALValue: AALDataPoint.value,
        });
      }
    });

    this.AMDData.forEach(AMDDataPoint => {
      const index = combinedDataWithNull.findIndex(data => data.date === AMDDataPoint.date);
      if (index !== -1) {
        // If date exists in combinedData, update AALValue
        combinedDataWithNull[index].AMDValue = AMDDataPoint.value;
      } else {
        // If date doesn't exist in combinedData, add a new point
        combinedDataWithNull.push({
          date: AMDDataPoint.date,
          AMDValue: AMDDataPoint.value,
        });
      }
    });

    this.AAPLData.forEach(AAPLDataPoint => {
      const index = combinedDataWithNull.findIndex(data => data.date === AAPLDataPoint.date);
      if (index !== -1) {
        // If date exists in combinedData, update AALValue
        combinedDataWithNull[index].AAPLValue = AAPLDataPoint.value;
      } else {
        // If date doesn't exist in combinedData, add a new point
        combinedDataWithNull.push({
          date: AAPLDataPoint.date,
          AAPLValue: AAPLDataPoint.value,
        });
      }
    });

        this.NVDAData.forEach(NVDADataPoint => {
      const index = combinedDataWithNull.findIndex(data => data.date === NVDADataPoint.date);
      if (index !== -1) {
        // If date exists in combinedData, update AALValue
        combinedDataWithNull[index].NVDAValue = NVDADataPoint.value;
      } else {
        // If date doesn't exist in combinedData, add a new point
        combinedDataWithNull.push({
          date: NVDADataPoint.date,
          NVDAValue: NVDADataPoint.value,
        });
      }
    });

    this.SnapData.forEach(SnapDataPoint => {
      const index = combinedDataWithNull.findIndex(data => data.date === SnapDataPoint.date);
      if (index !== -1) {
        // If date exists in combinedData, update AALValue
        combinedDataWithNull[index].SnapValue = SnapDataPoint.value;
      } else {
        // If date doesn't exist in combinedData, add a new point
        combinedDataWithNull.push({
          date: SnapDataPoint.date,
          SnapValue: SnapDataPoint.value,
        });
      }
    });

    this.SQData.forEach(SQDataPoint => {
      const index = combinedDataWithNull.findIndex(data => data.date === SQDataPoint.date);
      if (index !== -1) {
        // If date exists in combinedData, update AALValue
        combinedDataWithNull[index].SQValue = SQDataPoint.value;
      } else {
        // If date doesn't exist in combinedData, add a new point
        combinedDataWithNull.push({
          date: SQDataPoint.date,
          SQValue: SQDataPoint.value,
        });
      }
    });

    this.TSLAData.forEach(TSLADataPoint => {
      const index = combinedDataWithNull.findIndex(data => data.date === TSLADataPoint.date);
      if (index !== -1) {
        // If date exists in combinedData, update AALValue
        combinedDataWithNull[index].TSLAValue = TSLADataPoint.value;
      } else {
        // If date doesn't exist in combinedData, add a new point
        combinedDataWithNull.push({
          date: TSLADataPoint.date,
          TSLAValue: TSLADataPoint.value,
        });
      }
    });

        this.CData.forEach(CDataPoint => {
      const index = combinedDataWithNull.findIndex(data => data.date === CDataPoint.date);
      if (index !== -1) {
        // If date exists in combinedData, update AALValue
        combinedDataWithNull[index].CValue = CDataPoint.value;
      } else {
        // If date doesn't exist in combinedData, add a new point
        combinedDataWithNull.push({
          date: CDataPoint.date,
          CValue: CDataPoint.value,
        });
      }
    });

    // Set the result to combinedData
    this.combinedData = combinedDataWithNull;

    console.log("combinedData");
    console.log(this.combinedData);
  },

    async hideAll() {
      this.showAAPLLine = false;
      this.showAALLine = false;
      this.showAMDLine = false;
      this.showCLine = false;
      this.showNVDALine = false;
      this.showSnapLine = false;
      this.showSQLine = false;
      this.showTSLALine = false;
     },

    async showAll() {
      this.showAALLine = true;
      this.showCLine = true;
      this.showNVDALine = true;
      this.showSnapLine = true;
      this.showSQLine = true;
      this.showTSLALine = true;
      this.showAMDLine = true;
      this.showAAPLLine = true;
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
    
    async load_C_data() {
      try {
        const response = await axios.get(`${this.store.API}/predict/C`, {
          params: {
            stock_symbol: "AAPL"
          }
        });

        console.log("Prediction loaded");
        console.log(response.data);

        // Assuming the response.data is an object with date and close properties
        this.CData = Object.entries(response.data).map(([date, { value }]) => ({ date, value }));

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

    async load_AAPL_data() {
      try {
        const response = await axios.get(`${this.store.API}/predict/AAPL`, {
          params: {
            stock_symbol: "AAPL"
          }
        });

        console.log("Prediction loaded");
        console.log(response.data);

        // Assuming the response.data is an object with date and close properties
        this.AAPLData = Object.entries(response.data).map(([date, { value }]) => ({ date, value }));

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

    async load_AMD_data() {
      try {
        const response = await axios.get(`${this.store.API}/predict/AMD`, {
          params: {
            stock_symbol: "AAPL"
          }
        });

        // Assuming the response.data is an object with date and close properties
        this.AMDData = Object.entries(response.data).map(([date, { value }]) => ({ date, value }));

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

    async load_AAL_data() {
      try {
        const response = await axios.get(`${this.store.API}/predict/AAL`, {
          params: {
            stock_symbol: "AAPL"
          }
        });

        console.log("Prediction loaded");
        console.log(response.data);

        // Assuming the response.data is an object with date and close properties
        this.AALData = Object.entries(response.data).map(([date, { value }]) => ({ date, value }));

        console.log("mapped");
        console.log(this.AALData);

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

    async load_NVDA_data() {
      try {
        const response = await axios.get(`${this.store.API}/predict/NVDA`, {
          params: {
            stock_symbol: "AAPL"
          }
        });

        // Assuming the response.data is an object with date and close properties
        this.NVDAData = Object.entries(response.data).map(([date, { value }]) => ({ date, value }));

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

    async load_Snap_data() {
      try {
        const response = await axios.get(`${this.store.API}/predict/Snap`, {
          params: {
            stock_symbol: "AAPL"
          }
        });

        // Assuming the response.data is an object with date and close properties
        this.SnapData = Object.entries(response.data).map(([date, { value }]) => ({ date, value }));

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

    async load_SQ_data() {
      try {
        const response = await axios.get(`${this.store.API}/predict/SQ`, {
          params: {
            stock_symbol: "AAPL"
          }
        });

        // Assuming the response.data is an object with date and close properties
        this.SQData = Object.entries(response.data).map(([date, { value }]) => ({ date, value }));

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

    async load_TSLA_data() {
      try {
        const response = await axios.get(`${this.store.API}/predict/TSLA`, {
          params: {
            stock_symbol: "AAPL"
          }
        });

        // Assuming the response.data is an object with date and close properties
        this.TSLAData = Object.entries(response.data).map(([date, { value }]) => ({ date, value }));

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
      this.AALData = await this.load_AAL_data();
      this.CData = await this.load_C_data();
      this.NVDAData = await this.load_NVDA_data();
      this.SnapData = await this.load_Snap_data();
      this.SQData = await this.load_SQ_data();
      this.TSLAData = await this.load_TSLA_data();
      this.AMDData = await this.load_AMD_data();
      this.AAPLData = await this.load_AAPL_data();
      

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
  height: 30px;
  margin-left: 5px;
}

.button-stock:hover {
  background-color: darkgrey;
}
</style>