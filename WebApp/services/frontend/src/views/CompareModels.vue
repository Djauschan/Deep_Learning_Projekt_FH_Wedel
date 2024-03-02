<template>
    <Header />
  <div class="center">
    <!--<input class="text-input" v-model="selectedStock" placeholder="Please enter a stock">
    <input type="number" class="number-input" v-model.number="selectedDays" placeholder="Last n days">
    <button class="button-stock" @click="updateChart">Show Stock</button>!-->
    <span class="selection">Stock</span>
    <div class="selector">
      <select ref="selectorIn" @mouseover="changeCursor" @mouseleave="resetCursor">
        <option value="Option 1" selected>
          Apple
        </option>
        <option value="Option 2">
          American Airlines
        </option>
        <option value="Option 3">
          Advanced Micro Devices
        </option>
        <option value="Option 3">
          Citigroup
        </option>
        <option value="Option 3">
          NVIDIA
        </option>
        <option value="Option 3">
          Snap
        </option>
        <option value="Option 3">
          Block
        </option>
        <option value="Option 3">
          Tesla
        </option>
      </select>
    </div>
    <span class="selection">Time Interval</span>
    <div class="selector">
      <select ref="selectorIn" @mouseover="changeCursor" @mouseleave="resetCursor">
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
        value-field="CNNValue" argument-field="date">
      </DxSeries>
      <DxSeries v-if="showTransformerLine" :name="'Transformer' + selectedStock" :data-source="combinedData"
        type="line" value-field="TransformerValue" argument-field="date">
      </DxSeries>
      <DxSeries v-if="showLSTMLine" :name="'LSTM' + selectedStock" :data-source="combinedData" type="line"
        value-field="LSTMValue" argument-field="date">
      </DxSeries>
      <DxSeries v-if="showrandomForestLine" :name="'randomForest' + selectedStock"
        :data-source="combinedData" type="line" value-field="randomForestValue" argument-field="date">
      </DxSeries>
      <DxSeries v-if="showgradientBoostLine" :name="'gradientBoost' + selectedStock"
        :data-source="combinedData" type="line" value-field="gradientBoostValue" argument-field="date">
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
      <DxSeries :name="'CNN Line'" value-field="value" argument-field="date" type="line">
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
    <DxSeries :name="'Transformer Line'" value-field="value" argument-field="date" type="line">
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
      <DxSeries :name="'LSTM Line'" value-field="value" argument-field="date" type="line">
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
      <DxSeries :name="'randomForest Line'" value-field="value" argument-field="date" type="line">
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
      <DxSeries :name="'gradientBoost Line'" value-field="value" argument-field="date" type="line">
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


export default {
  components: {
  },
  async created() {
    this.rlData = [];
  },
  data() {
    return {
      rlData: [],
    };
  },

  async test() {
    console.log("test");
    this.rlData = await this.load_rl_data();
  },

    async load_reinforcementLearning_data() {
      console.log("Header");
      try {
        const response = await axios.get(`${this.store.API}/predict/rl`, {
          params: {
            stock_symbol: "AAPL"
          }
        });

        // Assuming the response.data is an object with date and close properties
        this.rlData = Object.entries(response.data).map(([date, { value }]) => ({ date, value }));

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
  }
  
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