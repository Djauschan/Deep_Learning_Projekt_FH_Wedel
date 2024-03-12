<template>
  <Header />
  <div class="home-container01">
    <span class="home-logo">Compare Stocks!</span>
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
              :class="{ 'selected-start': day === startDate }" :style="{ color: !isWeekday(day) ? 'red' : 'black' }"
              @click="selectDay(day)">
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
    <br>
    <div class="separator"></div>
    <span class="selection">Model</span>
    <div class="selector">
      <select v-model="selectedModel" ref="selectorIn" @mouseover="changeCursor" @mouseleave="resetCursor">
        <option value="transformer" selected>
          Transformer
        </option>
        <option value="cnn">
          CNN
        </option>
        <option value="randomForest">
          Random Forest
        </option>
        <option value="gradientBoost">
          Gradient Boost
        </option>
        <option value="lstm">
          LSTM
        </option>
      </select>
    </div>
    <div class="separator"></div>
    <button class="button-stock" @click="showAll">Check All</button>
    <br>
    <button class="button-stock" @click="hideAll">Uncheck All</button>
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
          <input v-if="selectedModel != 'transformer'" type="checkbox" v-model="showNIOLine"> NIO
        </label>
        <label>
          <input type="checkbox" v-model="showNVDALine"> NVDIA
        </label>
      </div>
      <div class="checkboxes3">
        <label>
          <input type="checkbox" v-model="showSNAPLine"> Snap
        </label>
        <label>
          <input type="checkbox" v-model="showSQLine"> Block
        </label>
        <label>
          <input type="checkbox" v-model="showTSLALine"> Tesla
        </label>
      </div>
      <br>
    </div>
    <div class="separator"></div>
    <button class="button-stock" @click="updateChart">Show Predictions</button>
  </div>
  <div v-if="isLoading" class="loading-container">
    {{ loadingMessage }}
  </div>
  <div class="newChart">
    <DxChart v-if="showAAPLLine && showChart" id="AAPL-chart" title="AAPL Chart"
      :data-source="this.combinedData['AAPL']">
      <!--<DxCommonSeriesSettings argument-field="DateTime" type="line" />-->
      <DxSeries :name="'AAPL Prediction'" value-field="value_pred" argument-field="DateTime" type="line"
        :color="seriesColors[8]">
      </DxSeries>
      <DxSeries :name="'AAPL Real Value'" value-field="Close" argument-field="DateTime" type="line"
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
    <DxChart v-if="showAALLine && showChart" id="AAL-chart" :data-source="this.combinedData['AAL']" title="AAL Chart">
      <DxCommonSeriesSettings argument-field="DateTime" type="line" />
      <DxSeries :name="'AAL Prediction'" value-field="value_pred" argument-field="DateTime" type="line" :color="seriesColors[8]">
      </DxSeries>
      <DxSeries :name="'AAL Real Value'" value-field="Close" argument-field="DateTime" type="line" :color="seriesColors[3]">
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
    <DxChart v-if="showAMDLine && showChart" id="AMD-chart" :data-source="this.combinedData['AMD']" title="AMD Chart">
      <DxCommonSeriesSettings argument-field="DateTime" type="line" />
      <DxSeries :name="'AMD Prediction'" value-field="value_pred" argument-field="DateTime" type="line" :color="seriesColors[8]">
      </DxSeries>
      <DxSeries :name="'AMD Real Value'" value-field="Close" argument-field="DateTime" type="line" :color="seriesColors[3]">
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
    <DxChart v-if="showCLine && showChart" id="C-chart" :data-source="this.combinedData['C']" title="C Chart">
      <DxCommonSeriesSettings argument-field="DateTime" type="line" />
      <DxSeries :name="'C Prediction'" value-field="value_pred" argument-field="DateTime" type="line" :color="seriesColors[8]">
      </DxSeries>
      <DxSeries :name="'C Real Value'" value-field="Close" argument-field="DateTime" type="line" :color="seriesColors[3]">
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
    <DxChart v-if="showNIOLine && showChart" id="NIO-chart" :data-source="this.combinedData['NIO']" title="NIO Chart">
      <DxCommonSeriesSettings argument-field="DateTime" type="line" />
      <DxSeries :name="'NIO Prediction'" value-field="value_pred" argument-field="DateTime" type="line" :color="seriesColors[8]">
      </DxSeries>
      <DxSeries :name="'NIO Real Value'" value-field="Close" argument-field="DateTime" type="line" :color="seriesColors[3]">
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
    <DxChart v-if="showNVDALine && showChart" id="NVDA-chart" :data-source="this.combinedData['NVDA']"
      title="NVDA Chart">
      <DxCommonSeriesSettings argument-field="DateTime" type="line" />
      <DxSeries :name="'NVDA Prediction'" value-field="value_pred" argument-field="DateTime" type="line"
        :color="seriesColors[8]">
      </DxSeries>
      <DxSeries :name="'NVDA Real Value'" value-field="Close" argument-field="DateTime" type="line"
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
    <DxChart v-if="showSNAPLine && showChart" id="SNAP Average-chart" :data-source="this.combinedData['SNAP']"
      title="SNAP Average Chart">
      <DxCommonSeriesSettings argument-field="DateTime" type="line" />
      <DxSeries :name="'Snap Prediction'" value-field="value_pred" argument-field="DateTime" type="line"
        :color="seriesColors[8]">
      </DxSeries>
      <DxSeries :name="'Snap Real Value'" value-field="Close" argument-field="DateTime" type="line"
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
    <DxChart v-if="showSQLine && showChart" id="SQ-chart" :data-source="this.combinedData['SQ']" title="SQ Chart">
      <DxCommonSeriesSettings argument-field="DateTime" type="line" />
      <DxSeries :name="'SQ Prediction'" value-field="value_pred" argument-field="DateTime" type="line" :color="seriesColors[8]">
      </DxSeries>
      <DxSeries :name="'SQ Real Value'" value-field="Close" argument-field="DateTime" type="line" :color="seriesColors[3]">
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
    <DxChart v-if="showTSLALine && showChart" id="TSLA-chart" :data-source="this.combinedData['TSLA']"
      title="TSLA Chart">
      <DxCommonSeriesSettings argument-field="DateTime" type="line" />
      <DxSeries :name="'TSLA Prediction'" value-field="value_pred" argument-field="DateTime" type="line"
        :color="seriesColors[8]">
      </DxSeries>
      <DxSeries :name="'TSLA Real Value'" value-field="Close" argument-field="DateTime" type="line"
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
    this.selectedDays = null;
  },
  data() {
    return {
      activeCharts: [],
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
      showSNAPLine: false,
      SNAPpredictionData: [],
      showSQLine: false,
      SQpredictionData: [],
      showTSLALine: false,
      TSLApredictionData: [],
      showNIOLine: false,
      NIOPredictionData: [],
      combinedData: [],
      store: useMyPiniaStore(),
      selectedTime: 'H',
      selectedModel: 'transformer',
      seriesColors: ['#FF5733', '#33FF57', '#337AFF', '#FF33DC', '#33FFDC', '#FFB733', '#FF3385', '#33B5FF'], // Array of colors for each series
      showCalendar: false,
      currentDate: new Date("2021-01-04"), // Current date
      currentMonth: '', // Current month displayed in the header
      currentMonthName: '', // Current month name displayed in the header
      currentYear: '', // Current year displayed in the header
      daysInMonth: [], // Array to hold days of the month
      startDate: null, // Selected start date
      selectingStart: true, // Flag to indicate if currently selecting start date
      isLoading: false,
      loadingMessage: 'Loading',
      loadingInterval: null,
      minDate: new Date(2021, 0, 4), // January is 0 in JavaScript
      maxDate: new Date(2021, 1, 10),
    };
  },

  watch: {
    currentDate() {
      this.getDaysInMonth();
      this.updateCurrentMonth();
    },
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
    formattedStartDate() {
      return this.startDate ? this.formatDate(this.startDate) : '';
    },

    tooltip: {
      enabled: true,
      // Customize tooltip appearance or behavior if needed
    },
  },

  mounted() {
    this.handleSelection();
    this.getDaysInMonth();
    this.updateCalendar();
  },

  methods: {
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
    formatCalendarEntry(dateString) {
      let date = new Date(dateString);

      let year = date.getFullYear();
      let month = date.getMonth() + 1;
      let day = date.getDate();

      // Pad single digit numbers with a leading zero
      month = month < 10 ? '0' + month : month;
      day = day < 10 ? '0' + day : day;

      // Return the formatted date string
      return `${year}-${month}-${day}`;
    },
    toggleCalendar() {
      this.startDate = null;
      this.selectingStart = true;
      this.showCalendar = !this.showCalendar;
    },
    previousMonth() {
      // Logic for moving to the previous month
      this.currentDate.setMonth(this.currentDate.getMonth() - 1);
      this.updateCalendar();
    },
    nextMonth() {
      // Logic for moving to the next month
      this.currentDate.setMonth(this.currentDate.getMonth() + 1);
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
    calculateErrorsAndDisplay(actualValues, predictedValues) {
      const { meanError, meanAbsoluteError } = calculateMeanErrorAndMeanAbsoluteError(actualValues, predictedValues);
      this.meanError = meanError.toFixed(2); // Round to 2 decimal places
      this.meanAbsoluteError = meanAbsoluteError.toFixed(2); // Round to 2 decimal places
    },
    displayErrors() {
      // Example actual and predicted data
      const actualValues = [10, 20, 30, 40, 50];
      const predictedValues = [12, 18, 28, 38, 48];
      this.calculateErrorsAndDisplay(actualValues, predictedValues);
    },
    handleSelection() {
    },
    async hideAll() {
      this.showAAPLLine = false;
      this.showAALLine = false;
      this.showAMDLine = false;
      this.showCLine = false;
      this.showNVDALine = false;
      this.showSNAPLine = false;
      this.showSQLine = false;
      this.showTSLALine = false;
      this.showNIOLine = false;
    },
    async showAll() {
      this.showAALLine = true;
      this.showCLine = true;
      this.showNVDALine = true;
      this.showSNAPLine = true;
      this.showSQLine = true;
      this.showTSLALine = true;
      this.showAMDLine = true;
      this.showAAPLLine = true;
      if (this.selectedModel != 'transformer') {
        this.showNIOLine = true;
      }
    },
    async load_model_data() {
      console.log(this.selectedModel)
      console.log("[" + this.activeCharts.join(', ') + "]")

      try {
        const response = await axios.get(`${this.store.API}/get/combinedData`, {
          params: {
            stock_symbols: "[" + this.activeCharts.join(', ') + "]",
            start_date: this.formatCalendarEntry(this.formattedStartDate),
            resolution: this.selectedTime,
            modelType: this.selectedModel,
          }
        });

        console.log("combinedData loaded");
        console.log(response.data);

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
          Swal.fire({
            title: "Error",
            text: "Please select a date between 2021-01-04 and 2021-02-10",
            icon: "info",
            showCloseButton: false,
            confirmButtonText: "Close",
            confirmButtonColor: "#d0342c",
          });
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
    checkDataInput() {
      console.log(this.activeCharts.length);
      console.log(this.formattedStartDate);
      if (this.showAALLine) {
        this.activeCharts.push('AAL');
      }
      if (this.showCLine) {
        this.activeCharts.push('C');
      }
      if (this.showNIOLine) {
        this.activeCharts.push('NIO');
      }
      if (this.showNVDALine) {
        this.activeCharts.push('NVDA');
      }
      if (this.showSNAPLine) {
        this.activeCharts.push('SNAP');
      }
      if (this.showSQLine) {
        this.activeCharts.push('SQ');
      }
      if (this.showTSLALine) {
        this.activeCharts.push('TSLA');
      }
      if (this.showAMDLine) {
        this.activeCharts.push('AMD');
      }
      if (this.showAAPLLine) {
        this.activeCharts.push('AAPL');
      }
      if (this.activeCharts.length > 0 && this.formattedStartDate != "") {
        return true;
      } else {
        return false;
      }
    },
    async updateChart() {
      if (this.checkDataInput()) {
        this.showChart = false;
        this.isLoading = true;
        this.startLoadingAnimation();
        this.combinedData = await this.load_model_data();

        console.log("this.combinedData")
        console.log(this.combinedData)

        this.showChart = true;
        this.isLoading = false;
        this.stopLoadingAnimation();
        this.activeCharts = [];
      } else {
        Swal.fire({
          title: "Error",
          text: "Please select a date, and at least one stock to display",
          icon: "info",
          showCloseButton: false,
          confirmButtonText: "Close",
          confirmButtonColor: "#d0342c",
        });
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

.loading-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 50vh;
  /* Adjust as needed */
  font-size: 1.5em;
  /* Adjust as needed */
}
</style>