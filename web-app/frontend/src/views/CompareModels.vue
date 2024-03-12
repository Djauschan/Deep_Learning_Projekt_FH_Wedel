<template>
  <Header />
  <div class="home-container01">
    <span class="home-logo">Compare Models!</span>
  </div>
  <div class="center">
    <div>
      <button class="button-stock" @click="toggleCalendar">Select Dates</button>
      <div class="calendar-popup" v-if="showCalendar">
        <div class="calendar">
          <div class="calendar-header">
            <button @click="previousMonth">&lt;</button>
            <div>
              <h2>{{ currentMonthName }}</h2>
              <h2>{{ currentMonth }}</h2>
              <h2>{{ currentYear }}</h2>
            </div>
            <button @click="nextMonth">&gt;</button>
          </div>
          <div class="calendar-days">
            <div v-for="day in daysInMonth" :key="day" class="calendar-day"
              :class="{ 'selected-start': day === startDate }" @click="selectDay(day)">
              {{ day }}
            </div>
          </div>
        </div>
      </div>
      <div>
        <label>Selected Start Date:</label>
      </div>
      <div>
        <input type="text" v-model="formattedStartDate" readonly class="short-input">
      </div>
    </div>

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
    <div class="separator"></div>
    <button class="button-stock" @click="updateChart">Load Charts</button>
  </div>
  <div v-if="isLoading" class="loading-container">
    {{ loadingMessage }}
  </div>
  <!-- Combined Chart -->
  <div class="newChart">
    <DxChart v-if="showChart" :data-source="combinedData" title="Stock Price">
      <DxCommonSeriesSettings argument-field="DateTime" type="stock" />
      <DxSeries :name=selectedStock open-value-field="Open" high-value-field="High" low-value-field="Low"
        close-value-field="Close" argument-field="DateTime">
      </DxSeries>
      <DxSeries v-if="showCNNLine" :name="'CNN-' + this.selectedStock" :data-source="combinedData" type="line"
        value-field="CNNValue" argument-field="date" :color="seriesColors[0]">
      </DxSeries>
      <DxSeries v-if="showTransformerLine" :name="'Transformer-' + this.selectedStock" :data-source="combinedData"
        type="line" value-field="TransformerValue" argument-field="date" :color="seriesColors[1]">
      </DxSeries>
      <DxSeries v-if="showLSTMLine" :name="'LSTM-' + this.selectedStock" :data-source="combinedData" type="line"
        value-field="LSTMValue" argument-field="date" :color="seriesColors[2]">
      </DxSeries>
      <DxSeries v-if="showrandomForestLine" :name="'randomForest-' + this.selectedStock" :data-source="combinedData"
        type="line" value-field="randomForestValue" argument-field="date" :color="seriesColors[3]">
      </DxSeries>
      <DxSeries v-if="showgradientBoostLine" :name="'gradientBoost-' + this.selectedStock" :data-source="combinedData"
        type="line" value-field="gradientBoostValue" argument-field="date" :color="seriesColors[4]">
      </DxSeries>
      <DxArgumentAxis :workdays-only="true">
        <DxTitle text="Time" />
        <DxLabel format="shortDate" />
      </DxArgumentAxis>
      <DxValueAxis name="price" position="left" :min="priceRange.min * 0.9">
        <DxTitle text="US dollars" />
        <DxLabel>
          <DxFormat type="currency" precision="2" />
        </DxLabel>
      </DxValueAxis>
      <DxTooltip :enabled="true" />
    </DxChart>
  </div>
  <div class="newChart">
    <DxChart v-if="showCNNLine && showChart" id="CNN-chart" :data-source="this.CNNData[this.selectedStock]"
      :title="CNNchartTitle">
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
          <DxFormat type="currency" precision="2" />
        </DxLabel>
      </DxValueAxis>
      <DxTooltip :enabled="true" />
    </DxChart>
  </div>
  <div class="newChart">
    <DxChart v-if="showTransformerLine && showChart" id="Transformer-chart"
      :data-source="this.transformerData[this.selectedStock]" :title="TransformerchartTitle">
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
          <DxFormat type="currency" precision="2" />
        </DxLabel>
      </DxValueAxis>
      <DxTooltip :enabled="true" />
    </DxChart>
  </div>
  <div class="newChart">
    <DxChart v-if="showLSTMLine && showChart" id="LSTM-chart" :data-source="this.LSTMData"
      :title="LSTMchartTitle[this.selectedStock]">
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
          <DxFormat type="currency" precision="2" />
        </DxLabel>
      </DxValueAxis>
      <DxTooltip :enabled="true" />
    </DxChart>
  </div>
  <div class="newChart">
    <DxChart v-if="showrandomForestLine && showChart" id="Random Forest-chart"
      :data-source="this.randomForestData[this.selectedStock]" :title="RandomForestchartTitle">
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
          <DxFormat type="currency" precision="2" />
        </DxLabel>
      </DxValueAxis>
      <DxTooltip :enabled="true" />
    </DxChart>
  </div>
  <div class="newChart">
    <DxChart v-if="showgradientBoostLine && showChart" id="Gradient Boost-chart"
      :data-source="this.gradientBoostData[this.selectedStock]" :title="GradientBoostchartTitle">
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
          <DxFormat type="currency" precision="2" />
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
    this.real_data = [];
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
      selectedStock: "AAL",
      seriesColors: ['#FF5733', '#33FF57', '#337AFF', '#FF33DC', '#33FFDC'], // Array of colors for each series
      cnnMeanError: null,
      cnnMeanAbsoluteError: null,
      cnnProfit: null,
      tfMeanError: null,
      tfMeanAbsoluteError: null,
      tfProfit: null,
      lstmMeanError: null,
      lstmMeanAbsoluteError: null,
      lstmProfit: null,
      gbMeanError: null,
      gbMeanAbsoluteError: null,
      gbProfit: null,
      rfMeanError: null,
      rfMeanAbsoluteError: null,
      rfProfit: null,
      showCalendar: false,
      currentDate: new Date("2021-01-04"), // Current date
      currentMonth: '', // Current month displayed in the header
      currentMonthName: '', // Current month name displayed in the header
      currentYear: '', // Current year displayed in the header
      daysInMonth: [], // Array to hold days of the month
      startDate: null, // Selected start date
      selectingStart: true, // Flag to indicate if currently selecting start date
      minDate: new Date(2021, 0, 4), // January is 0 in JavaScript
      maxDate: new Date(2021, 1, 10),
      isLoading: false,
      loadingMessage: 'Loading',
      loadingInterval: null,
    };
  },

  watch: {
    currentDate() {
      this.getDaysInMonth();
      this.updateCurrentMonth();
    },
  },

  computed: {

    currentMonth() {
      const months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
      let currentMonthIndex = this.currentDate.getMonth() + (this.currentMonthOffset - this.currentMonthOffset);

      // Adjust the index to handle cases where it goes beyond the boundaries
      if (currentMonthIndex < 0) {
        currentMonthIndex += 12; // Wrap around to December
      } else if (currentMonthIndex >= 12) {
        currentMonthIndex -= 12; // Wrap around to January
      }

      return months[currentMonthIndex];
    },

    CNNchartTitle() {
      return `CNN Chart; Mean Error: ${this.cnnMeanError}; Mean Absolute Error: ${this.cnnMeanAbsoluteError}; Gewinn: ${this.cnnProfit}$`;
    },
    TransformerchartTitle() {
      return `Transformer Chart; Mean Error: ${this.tfMeanError}; Mean Absolute Error: ${this.tfMeanAbsoluteError}; Gewinn: ${this.tfProfit}$`;
    },
    LSTMchartTitle() {
      return `LSTM Chart; Mean Error: ${this.lstmMeanError}; Mean Absolute Error: ${this.lstmMeanAbsoluteError}; Gewinn: ${this.lstmProfit}$`;
    },
    RandomForestchartTitle() {
      return `Random Forest Chart; Mean Error: ${this.rfMeanError}; Mean Absolute Error: ${this.rfMeanAbsoluteError}; Gewinn: ${this.rfProfit}$`;
    },
    GradientBoostchartTitle() {
      return `Gradient Boost Chart; Mean Error: ${this.gbMeanError}; Mean Absolute Error: ${this.gbMeanAbsoluteError}; Gewinn: ${this.gbProfit}$`;
    },
    tooltip: {
      enabled: true,
      // Customize tooltip appearance or behavior if needed
    },
    formattedStartDate() {
      return this.startDate ? this.formatDate(this.startDate) : '';
    },
  },

  mounted() {
    this.handleSelection();
    this.getDaysInMonth();
    this.updateCalendar();
  },

  methods: {
    updateCalendarPopup() {
      // Remove the color from the start day
      const startDayElement = document.querySelector('.selected-start');
      if (startDayElement) {
        startDayElement.classList.remove('selected-start');
      }
    },
    toggleCalendar() {
      this.startDate = null;
      this.selectingStart = true;
      this.showCalendar = !this.showCalendar;
    },
    previousMonth() {
      // Logic for moving to the previous month
      this.currentDate.setMonth(this.currentDate.getMonth() - 1);
      this.updateCalendarPopup();
      this.updateCalendar();
    },
    nextMonth() {
      // Logic for moving to the next month
      this.currentDate.setMonth(this.currentDate.getMonth() + 1);
      this.updateCalendarPopup();
      this.updateCalendar();
    },
    updateCalendar() {
      const year = this.currentDate.getFullYear();
      const month = this.currentDate.getMonth();
      const daysInMonth = new Date(year, month + 1, 0).getDate();
      const monthNames = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
      this.currentMonthName = monthNames[month];
      this.currentMonth = month + 1;
      this.currentYear = year;
      this.daysInMonth = Array.from({ length: daysInMonth }, (_, i) => i + 1);
    },
    formatDate(date) {
      const year = this.currentDate.getFullYear();
      const month = this.currentDate.getMonth();
      return new Date(year, month, date).toLocaleDateString('en-US');
    },



    getDaysInMonth() {
      const year = this.currentDate.getFullYear();
      const month = this.currentDate.getMonth();
      const daysInMonth = new Date(year, month + 1, 0).getDate();
      this.daysInMonth = Array.from({ length: daysInMonth }, (_, i) => i + 1);
    },

    selectDate(day) {
      const year = this.currentDate.getFullYear();
      const month = this.currentDate.getMonth() + 1; // Month is zero-based
      this.selectedDate = `${year}-${month.toString().padStart(2, '0')}-${day.toString().padStart(2, '0')}`;
      this.showDatePicker = false;
    },

    handleSelection() {
      //console.log("SelectedTime:", this.selectedTime);
      //console.log("SelectedStock:", this.selectedStock);
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
      if (this.showCNNLine) {
        mergeDataPoints(this.CNNData, 'CNNValue');
      }
      if (this.showTransformerLine) {
        mergeDataPoints(this.transformerData, 'TransformerValue');
      }
      if (this.showLSTMLine) {
        mergeDataPoints(this.LSTMData, 'LSTMValue');
      }
      if (this.showrandomForestLine) {
        mergeDataPoints(this.randomForestData, 'randomForestValue');
      }
      if (this.showgradientBoostLine) {
        mergeDataPoints(this.gradientBoostData, 'gradientBoostValue');
      }
      if (this.lstmLine) {
        mergeDataPoints(this.LSTMData, 'LSTMValue');
      }


      // Set the result to combinedData
      this.combinedData = combinedDataWithNull;

      console.log("combinedData");
      console.log(this.combinedData);
    },

    formatCalendarEntry(dateString) {
      // Create a new Date object
      let date = new Date(dateString);

      // Format the date
      let year = date.getFullYear();
      let month = date.getMonth() + 1; // Months are zero-based
      let day = date.getDate();

      // Pad single digit numbers with a leading zero
      month = month < 10 ? '0' + month : month;
      day = day < 10 ? '0' + day : day;

      // Return the formatted date string
      return `${year}-${month}-${day}`;
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

    async load_CNN_data() {
      try {
        const response = await axios.get(`${this.store.API}/predict/cnn`, {
          params: {
            stock_symbols: "[" + this.selectedStock + "]",
            start_date: this.formatCalendarEntry(this.formattedStartDate), // Replace with the start date of the prediction
            resolution: this.selectedTime, // Replace with the resolution:
          }
        });

        console.log("Prediction cnn loaded");
        console.log(response.data);
        if (response.data['status_code'] !== undefined) {
          this.showCNNLine = false;
        }

        // Assuming the response.data is an object with date and close properties
        //this.CNNData = Object.entries(response.data).map(([date, { value }]) => ({ date, value }));

        console.log("mapped cnnData:");
        console.log(this.CNNData);

        const maeResponse = await axios.get(`${this.store.API}/get/MAE`, {
          params: {
            stock_symbols: "[" + this.selectedStock + "]",
            start_date: this.formatCalendarEntry(this.formattedStartDate), // Replace with the start date of the prediction
            resolution: this.selectedTime,
            model_type: "cnn",
          }
        });
        this.cnnMeanError = maeResponse.data[this.selectedStock].ME;
        this.cnnMeanAbsoluteError = maeResponse.data[this.selectedStock].MAE;
        this.cnnProfit = maeResponse.data[this.selectedStock].profit;

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
            start_date: this.formatCalendarEntry(this.formattedStartDate), // Replace with the start date of the prediction
            resolution: this.selectedTime, // Replace with the resolution:
          }
        });

        console.log("Prediction transformer loaded");
        console.log(response.data);

        // Assuming the response.data is an object with date and close properties
        this.transformerData = Object.entries(response.data).map(([date, { value }]) => ({ date, value }));

        console.log("mapped transformerData:");
        console.log(this.transformerData);
        if (response.data['status_code'] !== undefined) {
          this.showTransformerLine = false;
        }

        const maeResponse = await axios.get(`${this.store.API}/get/MAE`, {
          params: {
            stock_symbols: "[" + this.selectedStock + "]",
            start_date: this.formatCalendarEntry(this.formattedStartDate), // Replace with the start date of the prediction
            resolution: this.selectedTime,
            model_type: "transformer",
          }
        });
        this.tfMeanError = maeResponse.data[this.selectedStock].ME;
        this.tfMeanAbsoluteError = maeResponse.data[this.selectedStock].MAE;
        this.tfProfit = maeResponse.data[this.selectedStock].profit;
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
            stock_symbols: "[" + this.selectedStock + "]",
            start_date: this.formatCalendarEntry(this.formattedStartDate), // Replace with the start date of the prediction
            resolution: this.selectedTime, // Replace with the resolution:
          }
        });

        console.log("Prediction randomforst loaded");
        console.log(response.data);
        if (response.data['status_code'] !== undefined) {
          this.showrandomForestLine = false;
        }

        // Assuming the response.data is an object with date and close properties
        //this.randomForestData = Object.entries(response.data).map(([date, { value }]) => ({ date, value }));

        console.log("mapped randomForestData:");
        console.log(this.randomForestData);
        const maeResponse = await axios.get(`${this.store.API}/get/MAE`, {
          params: {
            stock_symbols: "[" + this.selectedStock + "]",
            start_date: this.formatCalendarEntry(this.formattedStartDate), // Replace with the start date of the prediction
            resolution: this.selectedTime,
            model_type: "randomforest",
          }
        });
        this.rfMeanError = maeResponse.data[this.selectedStock].ME;
        this.rfMeanAbsoluteError = maeResponse.data[this.selectedStock].MAE;
        this.rfProfit = maeResponse.data[this.selectedStock].profit;
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
            stock_symbols: "[" + this.selectedStock + "]",
            start_date: this.formatCalendarEntry(this.formattedStartDate), // Replace with the start date of the prediction
            resolution: this.selectedTime, // Replace with the resolution:
          }
        });

        console.log("Prediction gradientBoost loaded");
        console.log(response.data);
        if (response.data['status_code'] !== undefined) {
          this.showgradientBoostLine = false;
        }

        // Assuming the response.data is an object with date and close properties
        //this.gradientBoostData = Object.entries(response.data).map(([date, { value }]) => ({ date, value }));

        console.log("mapped gradientBoostData:");
        console.log(this.gradientBoostData);
        const maeResponse = await axios.get(`${this.store.API}/get/MAE`, {
          params: {
            stock_symbols: "[" + this.selectedStock + "]",
            start_date: this.formatCalendarEntry(this.formattedStartDate), // Replace with the start date of the prediction
            resolution: this.selectedTime,
            model_type: "gradientboost",
          }
        });
        this.gbMeanError = maeResponse.data[this.selectedStock].ME;
        this.gbMeanAbsoluteError = maeResponse.data[this.selectedStock].MAE;
        this.gbProfit = maeResponse.data[this.selectedStock].profit;
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
            stock_symbols: "[" + this.selectedStock + "]",
            start_date: this.formatCalendarEntry(this.formattedStartDate), // Replace with the start date of the prediction
            resolution: this.selectedTime, // Replace with the resolution:
          }
        });

        console.log("Prediction lstm loaded");
        console.log(response.data);
        if (response.data['status_code'] !== undefined) {
          this.showLSTMLine = false;
        }
        const maeResponse = await axios.get(`${this.store.API}/get/MAE`, {
          params: {
            stock_symbols: "[" + this.selectedStock + "]",
            start_date: this.formatCalendarEntry(this.formattedStartDate), // Replace with the start date of the prediction
            resolution: this.selectedTime,
            model_type: "lstm",
          }
        });
        this.lstmMeanError = maeResponse.data[this.selectedStock].ME;
        this.lstmMeanAbsoluteError = maeResponse.data[this.selectedStock].MAE;
        this.lstmProfit = maeResponse.data[this.selectedStock].profit;

        // Assuming the response.data is an object with date and close properties
        //this.LSTMData = Object.entries(response.data).map(([date, { value }]) => ({ date, value }));

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
            start_date: this.formatCalendarEntry(this.formattedStartDate), // Replace with the start date of the prediction
            resolution: this.selectedTime,
          }
        });
        console.log("### normal loaded ###")
        if (response.data['status_code'] !== undefined) {
          return [];
        }
        else {
          console.log(response.data[this.selectedStock])
          return response.data[this.selectedStock]
        }
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
      this.showChart = false;
      this.isLoading = true;
      this.startLoadingAnimation();
      console.log("##### SelectedTime: ", this.selectedTime);
      console.log("##### SelectedStock:", this.selectedStock);
      console.log("##### checkAllBoxes:", this.checkAllBoxes());

      if (this.selectedStock && this.selectedTime && this.checkAllBoxes() && this.startDate) {
        this.dataSource = await this.load_data();
        this.real_data = await this.load_data();
        if (this.showCNNLine) {
          this.CNNData = await this.load_CNN_data();
        }
        if (this.showTransformerLine) {
          this.transformerData = await this.load_transformer_data();
        }
        if (this.showLSTMLine) {
          this.LSTMData = await this.load_LSTM_data();
        }
        if (this.showrandomForestLine) {
          this.randomForestData = await this.load_randomForest_data();
        }
        if (this.showgradientBoostLine) {
          this.gradientBoostData = await this.load_gradientBoost_data();
        }

        console.log("datasource: " + this.dataSource)


        if (this.dataSource) {
          const prices = this.dataSource.flatMap(data => [data.Open, data.Close]);
          console.log('dataSource:', this.dataSource);
          console.log('prices:', prices);

          let minIsNull = false;
          for (let price in prices) {
            if (prices[price] == 0) {
              minIsNull = true;
            }
          }
          if (minIsNull) {
            const sum = prices.reduce((a, b) => a + b, 0);
            const avg = sum / prices.length;
            this.priceRange = {
              min: avg * 0.5,
              max: Math.max(...prices) * 2
            };
            this.priceRangeKey = Math.random();
            console.log("price range: " + this.priceRange.min + " - " + this.priceRange.max)
          } else {
            this.priceRange = {
              min: Math.min(...prices) * 0.5,
              max: Math.max(...prices) * 2
            };
            this.priceRangeKey = Math.random();
            console.log("price range: " + this.priceRange.min + " - " + this.priceRange.max)
          }
        }

        await this.updateCombinedData();
        this.showChart = true;
        this.isLoading = false;
        this.stopLoadingAnimation();
      } else {
        Swal.fire({
          title: "Error at getting data",
          text: "Please select a stock, the time interval, the start date and a model",
          icon: "info",
          showCloseButton: false,
          confirmButtonText: "Close",
          confirmButtonColor: "#d0342c",
        });
        this.isLoading = false;
      }
    },

    startLoadingAnimation() {
      let dots = 0;
      this.loadingInterval = setInterval(() => {
        dots = (dots + 1) % 4;
        this.loadingMessage = 'Loading' + '.'.repeat(dots);
      }, 1000);
    },

    stopLoadingAnimation() {
      clearInterval(this.loadingInterval);
      this.loadingMessage = 'Loading';
    },

    async get_stock_data(stock_symbol, days_back) {
      try {
        const response = await axios.get(this.store.API + "/getStock", {
          params: {
            stock_symbol: this.selectedStock,
            start_date: this.formatCalendarEntry(this.startDate), // Replace with the start date of the prediction
            resolution: this.selectedTime, // Replace with the resolution:
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
    selectDay(day) {
      if (this.isWeekday(day)) {
        const selectedDate = new Date(this.currentDate.getFullYear(), this.currentDate.getMonth(), day);
        if (selectedDate >= this.minDate && selectedDate <= this.maxDate) {
          // Handle day selection
          if (this.selectingStart) {
            this.startDate = day;
            this.showCalendar = false;
          }
        } else {
          alert('Please select a date between 2021-01-04 and 2021-02-10');
        }
      } else {
        Swal.fire({
          title: "Error",
          text: "Please select a weekday",
          icon: "info",
          showCloseButton: false,
          confirmButtonText: "Close",
          confirmButtonColor: "#d0342c",
        });
      }
    },
    isWeekday(day) {
      const dayOfWeek = new Date(this.currentDate.getFullYear(), this.currentDate.getMonth(), day).getDay();
      return dayOfWeek !== 0 && dayOfWeek !== 6; // 0 is Sunday, 6 is Saturday
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
.short-input {
  width: 100px;
  /* Adjust width as needed */
}

.loading-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 50vh;
  /* Adjust as needed */
  font-size: 1.5em;
  /* Adjust as needed */
}

.calendar-popup {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  z-index: 1000;
  background-color: white;
  border: 1px solid #ccc;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
  padding: 20px;
}

.calendar {
  font-family: Arial, sans-serif;
}

.calendar-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.calendar-days {
  display: grid;
  grid-template-columns: repeat(7, 1fr);
  gap: 5px;
}

.calendar-day {
  padding: 10px;
  border: 1px solid #ccc;
  cursor: pointer;
}

.selected-start {
  background-color: #90EE90;
  /* Light green */
}

.selected-range {
  background-color: #ADD8E6;
  /* Light blue */
}

.calendar-day:hover {
  background-color: #f0f0f0;
}

.popup {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background-color: rgba(0, 0, 0, 0.5);
  width: 100%;
  height: 100%;
}

.popup-content {
  position: relative;
  background-color: #fff;
  border: 1px solid #ccc;
  padding: 10px;
  max-width: 400px;
  /* Adjust max-width as needed */
  max-height: 300px;
  /* Adjust max-height as needed */
  overflow-y: auto;
  /* Add scrollbar when content exceeds max-height */
}

.date-picker-popup {
  width: 100%;
  /* Make date picker content full width */
}

.time-picker-popup {
  width: 100%;
  /* Make time picker content full width */
}

.popup h3 {
  margin-top: 0;
  margin-bottom: 10px;
}

.popup button {
  margin-top: 10px;
}


.date-time-picker {
  display: flex;
}

.popup {
  display: flex;
  justify-content: center;
  align-items: center;
}

.popup-content {
  border-radius: 5px;
}

.time-wheel {
  display: flex;
  flex-wrap: wrap;
}

.hour,
.minute {
  flex: 1 0 20%;
  text-align: center;
  padding: 5px;
  border: 1px solid #ccc;
  cursor: pointer;
}

.hour:first-child,
.minute:first-child {
  border-top-left-radius: 5px;
}

.hour:last-child,
.minute:last-child {
  border-top-right-radius: 5px;
}

.hour:nth-last-child(1),
.minute:nth-last-child(1) {
  border-bottom-left-radius: 5px;
}

.hour:nth-last-child(2),
.minute:nth-last-child(2) {
  border-bottom-right-radius: 5px;
}

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