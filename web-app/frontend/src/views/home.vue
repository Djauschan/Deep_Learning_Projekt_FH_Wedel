<template>
  <Header />
  <div :class="{ 'dark-mode': isDarkMode }">
    <div class="home-container">
      <div class="home-header">
        <header data-thq="thq-navbar" class="home-navbar-interactive">
          <div class="home-container01">
            <span class="home-logo">Alles auf einen Blick!</span>
          </div>
          <div class="home-container02">
            <div data-thq="thq-navbar-nav" class="home-desktop-menu"></div>
          </div>
          <div data-thq="thq-burger-menu" class="home-burger-menu">
            <svg viewBox="0 0 1024 1024" class="home-icon02">
              <path
                d="M128 554.667h768c23.552 0 42.667-19.115 42.667-42.667s-19.115-42.667-42.667-42.667h-768c-23.552 0-42.667 19.115-42.667 42.667s19.115 42.667 42.667 42.667zM128 298.667h768c23.552 0 42.667-19.115 42.667-42.667s-19.115-42.667-42.667-42.667h-768c-23.552 0-42.667 19.115-42.667 42.667s19.115 42.667 42.667 42.667zM128 810.667h768c23.552 0 42.667-19.115 42.667-42.667s-19.115-42.667-42.667-42.667h-768c-23.552 0-42.667 19.115-42.667 42.667s19.115 42.667 42.667 42.667z">
              </path>
            </svg>
          </div>
        </header>
      </div>
      <div class="home-container05">
        <div class="home-container18">
          <div class="home-container19">
            <div class="home-container20">
              <div class="home-container21">
                <div class="home-container22">
                  <div class="hover-effect">
                    <div class="home-container23">
                      <span>Budget:</span>
                      <div class="home-container24">
                        <span>{{ budget }}</span>
                      </div>
                    </div>
                  </div>
                </div>
                <div class="home-container25">
                  <div class="seperator"></div>
                  <div class="home-container29">
                    <div class="hover-effect">
                      <div class="home-container26">
                        <span>Budget aufladen:</span>
                        <div class="middleInput">
                          <input type="text" placeholder="Wert in Euro" class="home-textinput" v-model="budgetInput" />
                        </div>
                        <div class="home-container27">
                          <button ref="myButton4" @mouseover="changeCursor" @mouseleave="resetCursor" type="button"
                            class="button" @click="update_budget">Aufladen</button>
                          <button ref="myButton3000" @mouseover="changeCursor" @mouseleave="resetCursor" type="button"
                            class="button" @click="update_budget2">Auszahlen</button>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                <div class="seperator"></div>
                <div class="hover-effect">
                  <div class="home-container71">
                    <div class="home-container33">
                      <div class="home-container34">
                        <span>Aktienwahl</span>
                      </div>
                      <div class="home-container35">
                        <select v-model="selectedInterval" ref="myButton3" @mouseover="changeCursor"
                          @mouseleave="resetCursor">
                          <option value="D" selected>
                            Täglich
                          </option>
                          <option value="H">
                            Stündlich
                          </option>
                          <option value="M">
                            Minütlich
                          </option>
                        </select>
                      </div>
                    </div>
                  </div>
                </div>
                <div class="seperator"></div>
                <div class="hover-effect">
                  <div class="home-container33">
                    <div class="home-container34">
                      <span>Los geht&apos;s!</span>
                    </div>
                    <div class="home-container35">
                      <svg viewBox="0 0 1024 1024" class="home-icon33">
                        <path
                          d="M480 64c-265.096 0-480 214.904-480 480s214.904 480 480 480c265.098 0 480-214.902 480-480s-214.902-480-480-480zM480 928c-212.078 0-384-171.922-384-384s171.922-384 384-384c212.076 0 384 171.922 384 384s-171.924 384-384 384z">
                        </path>
                        <path
                          d="M670.824 644.34c-15.27-8.884-34.862-3.708-43.75 11.57-17.256 29.662-49.088 48.090-83.074 48.090h-128c-41.716 0-77.286-26.754-90.496-64h154.496c17.672 0 32-14.326 32-32s-14.328-32-32-32h-160v-64h160c17.672 0 32-14.328 32-32s-14.328-32-32-32h-154.496c13.21-37.246 48.78-64 90.496-64h128c33.986 0 65.818 18.426 83.074 48.090 8.888 15.276 28.478 20.456 43.752 11.568 15.276-8.888 20.456-28.476 11.568-43.752-28.672-49.288-81.702-79.906-138.394-79.906h-128c-77.268 0-141.914 55.056-156.78 128h-35.22c-17.672 0-32 14.328-32 32s14.328 32 32 32h32v64h-32c-17.672 0-32 14.326-32 32s14.328 32 32 32h35.22c14.866 72.944 79.512 128 156.78 128h128c56.692 0 109.72-30.62 138.394-79.91 8.888-15.276 3.708-34.864-11.57-43.75z">
                        </path>
                      </svg>
                      <button ref="myButton2" @mouseover="changeCursor" @mouseleave="resetCursor" type="button"
                        class="button" @click="load_rl_data">Prognose</button>
                    </div>
                  </div>
                </div>
                <div class="seperator"></div>
                <div class="hover-effect">
                  <div class="home-container33">
                    <div class="home-container34">
                      <span>Gewinn: {{ rlData ? rlData.profit : 'TBD' }}</span>
                    </div>
                  </div>
                </div>
              </div>
              </div>
            <div class="home-container36">
              
              <div class="home-container37">
                <div class="home-container38">
                  <div class="home-container39">
                    <div v-if="isLoading" class="loading-container">
                      {{ loadingMessage }}
                    </div>
                    <DxChart v-if="rlData" id="chart" :data-source="transformedData" title="Kapitalaufstellung">
                      <DxCommonSeriesSettings argument-field="timestamp" type="stackedBar" />
                      <DxSeries value-field="AAPL" name="Apple" stack="assets" />
                      <DxSeries value-field="NVDA" name="Nvidia" stack="assets" />
                      <DxSeries value-field="AAL" name="American Airlines" stack="assets" />
                      <DxSeries value-field="AMD" name="AMD" stack="assets" />
                      <DxSeries value-field="SNAP" name="Snap inc" stack="assets" />
                      <DxSeries value-field="C" name="Citigroup" stack="assets" />
                      <DxSeries value-field="SQ" name="Block (SQ)" stack="assets" />
                      <DxSeries value-field="liquid_money" name="Liquide Mittel" stack="assets" />
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
                      <DxTooltip :enabled="true" location="edge" />
                    </DxChart>
                  </div>
                </div>
              </div>
              <div class="home-container40">
                <div class="home-container43">
                  <div class="home-container42" v-if="rlData">
                    <span>Trade Historie</span>
                  </div>
                  <div class="table-container" v-if="rlData">
                      <table>
                        <thead>
                          <tr>
                            <th>Uhrzeit</th>
                            <th v-for="stock in Object.keys(rlData.actions[Object.keys(rlData.actions)[0]].actions)">{{ stock }}</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr v-for="(dateData, date) in rlData.actions" :key="date">
                            <td>{{ new Date(date).toLocaleString() }}</td>
                            <td v-for="stock in Object.keys(dateData.actions)">
                              {{ dateData.actions[stock] }}
                            </td>
                          </tr>
                        </tbody>
                      </table>
                  </div>

                  <!--<div class="home-container12">
                    <ProgressBar ref="progressBar" />
                  </div>-->
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="home-container48"></div>
    </div>
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
import ProgressBar from "@/components/ProgressBar.vue";
import SearchBar from '@/components/SearchBar.vue';
import { useMyPiniaStore } from "../store.js";
import { defineComponent, computed, ref } from "vue";
import axios from 'axios';
import Swal from "sweetalert2/dist/sweetalert2.js";

export default {
  name: "Home",
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
    ProgressBar,
    SearchBar,
  },
  setup() {

    const isUserLoggedIn = computed(() => {
      return localStorage.logged_user !== null;
    });

    return {
      isUserLoggedIn,
      username: localStorage.logged_user,
      email: localStorage.logged_email,
    };
  },
  async created() {
    this.budget = await this.get_budget();
  },
  data() {
    return {
      dataSource,
      isDarkMode: false,
      budgetInput: null,
      store: useMyPiniaStore(),
      rlData: null,
      lastEnsembleValue: null, // Property to store the last ensemble value
      budget: null,
      username: localStorage.logged_user,
      email: localStorage.logged_email,
      isLoading: false,
      loadingMessage: "Loading",
      selectedInterval: "D",
    };
  },
  mounted() {
    // Scroll to a specific position on the page after it's loaded
    this.scrollToPosition(0, 500); // Example: scroll to the top of the page with animation duration of 500ms


    // Extract the last "ensemble" value
    this.extractLastEnsembleValue();
  },
  computed: {
  transformedData() {
    if (!this.rlData) {
      return [];
    }

    const timestamps = Object.keys(this.rlData.actions);
    return timestamps.map(timestamp => {
      const assets = this.rlData.actions[timestamp].assets;
      return {
        timestamp,
        ...assets
      };
    });
  }
},
  methods: {
    extractLastEnsembleValue() {
      // Assuming rlData is available in your component's data
      if (this.rlData) {
        const lastDate = Object.keys(this.rlData).pop(); // Get the last date entry
        this.lastEnsembleValue = this.rlData[lastDate].ensemble; // Extract the last "ensemble" value
      }
    },

    scrollToPosition(x, y, duration = 0) {
      const scrollOptions = {
        left: x,
        top: y,
        behavior: 'smooth' // Smooth scrolling behavior
      };
      setTimeout(() => {
        window.scrollTo(scrollOptions);
      }, duration);
    },

    formatStockData(data) {
      let formattedData = "";
      for (const [key, value] of Object.entries(data)) {
        formattedData += `${key}: ${value}\n`;
      }
      return formattedData;
    },


    async load_rl_data() {
      this.rlData = null;
      this.isLoading = true;
      this.startLoadingAnimation();
      try {
        const response = await axios.get(`${this.store.API}/simulate/rl`, { // Use store.state.API
          params: {
            stock_symbols: '[AAPL, AAL, SNAP, TSLA, NVDA, C, SQ, AMD]',
            start_date: '2021-01-04',
            resolution: this.selectedInterval,
            username: this.username,

          }
        });
        this.stopLoadingAnimation();
        this.isLoading = false;
        this.rlData = response.data;
      } catch (error) {
        console.error(error);
      }
    },
    async get_budget() {
      const username = this.username;
      try {
        const response = await axios.get(this.store.API + "/get_budget/" + username, {
          params: {
            username: username,
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
    async update_budget() {
      console.log(this.budget);
      try {
        const response = await axios.put(this.store.API + "/update_user/" + this.username, {

          username: this.username,
          email: this.username,
          budget: this.budgetInput,

        });
        console.log([response.data])
        this.budget = await this.get_budget();
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
    async update_budget2() {
      console.log(this.budget);
      try {
        const response = await axios.put(this.store.API + "/update_user/" + this.username, {

          username: this.username,
          email: this.username,
          budget: this.budgetInput * -1,

        });
        console.log([response.data])
        this.budget = await this.get_budget();
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
    changeCursor() {
      //this.$refs.myButton.style.cursor = 'pointer'; // or any other cursor value
      //this.$refs.myButton1.style.cursor = 'pointer'; // or any other cursor value
      this.$refs.myButton2.style.cursor = 'pointer'; // or any other cursor value
      this.$refs.myButton3.style.cursor = 'pointer'; // or any other cursor value
      this.$refs.myButton4.style.cursor = 'pointer'; // or any other cursor value
      this.$refs.myButton3000.style.cursor = 'pointer'; // or any other cursor value
      //this.$refs.myButton5.style.cursor = 'pointer'; // or any other cursor value
    },
    resetCursor() {
      //this.$refs.myButton.style.cursor = 'auto'; // reset to the default cursor
      //this.$refs.myButton1.style.cursor = 'auto'; // reset to the default cursor
      this.$refs.myButton2.style.cursor = 'auto'; // reset to the default cursor
      this.$refs.myButton3.style.cursor = 'auto'; // reset to the default cursor
      this.$refs.myButton4.style.cursor = 'auto'; // reset to the default cursor
      this.$refs.myButton3000.style.cursor = 'auto'; // reset to the default cursor
      //this.$refs.myButton5.style.cursor = 'auto'; // reset to the default cursor
    },
    toggleDarkMode() {
      this.isDarkMode = !this.isDarkMode;
    },
    checkAuth() {

    },
    logout() {
      localStorage.logout();
      this.$router.push("/login");
    },
    customizeTooltip(pointInfo) {
      return {
        text: `Open: $${pointInfo.openValue}<br/>
                Close: $${pointInfo.closeValue}<br/>
                High: $${pointInfo.highValue}<br/>
                Low: $${pointInfo.lowValue}<br/>`,
      };
    },
    simulateProgress() {
      let progress = 0;
      const progressBar = this.$refs.progressBar;

      const intervalId = setInterval(() => {
        progress += 20;
        progressBar.updateProgress(progress);

        if (progress >= 100) {
          clearInterval(intervalId);
        }
      }, 500);
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
  },

  metaInfo: {
    title: "frontend",
    meta: [
      {
        property: "og:title",
        content: "frontend",
      },
    ],
  },
};
</script>

<style scoped>
.home-container {
  width: 100%;
  display: flex;
  overflow: auto;
  align-items: center;
  display: flex;
  flex-direction: column;
  height: 100vh;
  /* Set initial height to 100% of viewport height */
  flex: 1;
}

.dark-mode {
  background-color: #333;
  /* Dark background color */
  color: #fff;
  /* Light text color for better contrast */
}



.loading-spinner {
  border: 4px solid #f3f3f3;
  /* Light gray border */
  border-top: 4px solid #3498db;
  /* Blue border for spinner animation */
  border-radius: 50%;
  width: 20px;
  height: 20px;
  animation: spin 1s linear infinite;
  /* Spinner animation */
}

@keyframes spin {
  0% {
    transform: rotate(0deg);
  }

  100% {
    transform: rotate(360deg);
  }
}

#chart {
  height: 100%;
}

.search {
  width: 100%;
  object-fit: cover;
  margin-left: 15%;
}

.spread {
  width: 100%;
  display: flex;
  align-items: center;
  flex-direction: row;
  justify-content: space-around;
  margin-left: 10%;
}

.home-image {
  width: 340px;
  object-fit: cover;
}

.home-navbar-interactive {
  width: 100%;
  display: flex;
  position: sticky;
  max-width: var(--dl-size-size-maxwidth);
  align-items: center;
  padding-top: var(--dl-space-space-oneandhalfunits);
  padding-left: var(--dl-space-space-threeunits);
  padding-right: var(--dl-space-space-threeunits);
  justify-content: space-between;
}

.home-container01 {
  flex: 0 0 auto;
  height: 80px;
  display: flex;
  align-items: flex-end;
  flex-direction: column;
  justify-content: center;
}

.home-logo {
  font-size: 65px;
  align-self: flex-start;
  font-family: "Lato";
  font-weight: bold;
  margin-left: var(--dl-space-space-unit);
  text-transform: uppercase;
}

.home-icon {
  width: 35px;
  height: 101px;
  align-self: flex-end;
  margin-right: var(--dl-space-space-halfunit);
}

.home-container02 {
  flex: 0 0 auto;
  width: auto;
  height: auto;
  display: flex;
  align-items: space-between;
  flex-direction: column;
}

.home-header {
  height: 90px;
}

.home-desktop-menu {
  flex: 1;
  display: flex;
  justify-content: space-between;
}

.home-burger-menu {
  display: none;
}

.home-icon02 {
  width: var(--dl-size-size-xsmall);
  cursor: pointer;
  height: var(--dl-size-size-xsmall);
}

.home-mobile-menu {
  top: 0px;
  left: 0px;
  width: 100%;
  height: 100vh;
  display: none;
  padding: 32px;
  z-index: 100;
  position: absolute;
  border-radius: 50px;
  flex-direction: column;
  justify-content: space-between;
}

.home-nav {
  display: flex;
  align-items: flex-start;
  flex-direction: column;
}

.home-top {
  width: 100%;
  display: flex;
  align-items: center;
  margin-bottom: var(--dl-space-space-threeunits);
  justify-content: space-between;
}

.home-logo1 {
  font-size: 2em;
  font-weight: bold;
}

.home-close-menu {
  display: flex;
  align-items: center;
  justify-content: center;
}


.home-links {
  flex: 0 0 auto;
  display: flex;
  align-self: flex-start;
  align-items: flex-start;
  flex-direction: column;
}




.home-buttons {
  display: flex;
  margin-top: var(--dl-space-space-unit);
  align-items: center;
  flex-direction: row;
  justify-content: space-between;
}

.home-login {
  border-color: #E0E0E0;
  border-width: 1px;
  margin-right: var(--dl-space-space-twounits);
  padding-left: 1.5rem;
  border-radius: 30px;
  padding-right: 1.5rem;
}


.home-icon06 {
  width: var(--dl-size-size-xsmall);
  height: var(--dl-size-size-xsmall);
  margin-right: var(--dl-space-space-twounits);
}

.home-icon08 {
  width: var(--dl-size-size-xsmall);
  height: var(--dl-size-size-xsmall);
  margin-right: var(--dl-space-space-twounits);
}

.home-icon10 {
  width: var(--dl-size-size-xsmall);
  height: var(--dl-size-size-xsmall);
}

.home-container05 {
  flex: 0 0 auto;
  width: 100%;
  height: 70vh;
  display: flex;
  align-items: flex-start;
  justify-content: center;
}

.home-container18 {
  flex: 0 0 auto;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  flex-direction: row;
  justify-content: flex-start;
}

.home-container19 {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: flex-start;
  flex-direction: column;
}

.home-container20 {
  flex: 0 0 auto;
  width: 100%;
  height: auto;
  display: flex;
  align-items: space-between;
  justify-content: space-between;
}

.home-container21 {
  padding-top: 1%;
  width: 100%;
  height: auto;
  display: flex;
  align-items: center;
  /* Center items along the cross axis */
  justify-content: space-between;
  /* Distribute items along the main axis with equal space */
}

.seperator {
  border: 1px solid #E0E0E0;
  height: 100%;
}

.home-container22 {
  flex: 0 0 auto;
  width: 140px;
  height: 100px;
  display: flex;
  align-items: center;
  margin-left: var(--dl-space-space-unit);
  flex-direction: column;
  justify-content: center;
}

.home-container23 {
  width: 200px;
  height: 27px;
  display: flex;
  align-items: center;
  flex-direction: column;
  justify-content: center;
  border-radius: 10px;
  /* Adjust the value to control the roundness of the corners */
}

.home-container24 {
  flex: 0 0 auto;
  width: 100%;
  height: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 10px;
  margin-top: 5px;
  /* Adjust the value to control the roundness of the corners */
}

.table-container {
  overflow-x: auto;
  /* Add horizontal scrollbar if content exceeds container width */
  max-width: 100%;
  /* Limit the maximum width of the container */
}

table {
  width: 100%;
  /* Ensure table takes up full width of its container */
  border-collapse: collapse;
  /* Collapse borders between cells */
}

th,
td {
  padding: 8px;
  /* Add padding to cells */
  text-align: left;
  /* Align text to the left within cells */
  border: 1px solid #ddd;
  /* Add border to cells */
}

th {
  background-color: #f2f2f2;
  /* Add background color to header cells */
}


.home-icon26 {
  width: 24px;
  height: 24px;
  margin-right: var(--dl-space-space-halfunit);
  border-radius: 10px;
  /* Adjust the value to control the roundness of the corners */
}

.home-container25 {
  width: 200px;
  height: 100px;
  display: flex;
  align-items: center;
  flex-direction: row;
  justify-content: center;
}

.home-container26 {
  width: 100%;
  height: 100px;
  display: flex;
  align-items: center;
  flex-direction: column;
  justify-content: center;
}

.home-container262 {
  width: 100%;
  height: 100px;
  display: flex;
  align-items: center;
  flex-direction: column;
  justify-content: center;
}

.home-container27 {
  margin-top: 25px;
  width: 200px;
  height: 27px;
  display: flex;
  align-items: center;
  flex-direction: column;
  justify-content: center;
  border-radius: 10px;
  /* Adjust the value to control the roundness of the corners */
}

.home-container28 {
  flex: 0 0 auto;
  width: 266px;
  height: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 10px;
  /* Adjust the value to control the roundness of the corners */
}

.home-icon28 {
  width: 24px;
  height: 24px;
  margin-right: var(--dl-space-space-halfunit);
  border-radius: 10px;
  /* Adjust the value to control the roundness of the corners */
}

.middleInput {
  justify-content: center;

}

.home-textinput {
  width: 174px;
  height: 24px;
  border-radius: 10px;
  /* Adjust the value to control the roundness of the corners */
}

.home-container29 {
  width: 234px;
  height: 100px;
  display: flex;
  align-items: center;
  flex-direction: row;
  justify-content: center;
}

.home-container30 {
  flex: 0 0 auto;
  width: 320px;
  height: 100px;
  display: flex;
  align-items: center;
  flex-direction: column;
  justify-content: center;
}

.home-container31 {
  width: 200px;
  height: 27px;
  display: flex;
  align-items: center;
  flex-direction: column;
  justify-content: center;
  margin-top: 5px;
  border-radius: 10px;
  /* Adjust the value to control the roundness of the corners */
}

.home-container32 {
  flex: 0 0 auto;
  width: 100%;
  height: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 10px;
  /* Adjust the value to control the roundness of the corners */
}

.home-container70 {
  flex: 0 0 auto;
  width: auto;
  height: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.home-container71 {
  flex: 0 0 auto;
  width: auto;
  height: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.home-icon30 {
  width: 50px;
  height: 50px;
  margin-right: var(--dl-space-space-halfunit);
  border-radius: 10px;
  /* Adjust the value to control the roundness of the corners */
}

.home-container33 {
  flex: 0 0 auto;
  width: 180px;
  height: 100px;
  display: flex;
  align-items: center;
  margin-right: var(--dl-space-space-unit);
  flex-direction: column;
  justify-content: center;
}

.home-container34 {
  width: 180px;
  height: 27px;
  display: flex;
  align-items: center;
  flex-direction: column;
  justify-content: center;
  border-radius: 10px;
  /* Adjust the value to control the roundness of the corners */
}

.home-container35 {
  flex: 0 0 auto;
  width: 266px;
  height: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 10px;
  /* Adjust the value to control the roundness of the corners */
}

.home-icon33 {
  width: 24px;
  height: 24px;
  margin-right: var(--dl-space-space-halfunit);
}

.home-container36 {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.home-container37 {
  flex: 0 0 auto;
  width: 50%;
  height: auto;
  display: flex;
  align-self: top;
  margin-top: 3%;
  align-items: center;
  margin-left: var(--dl-space-space-unit);
  margin-right: var(--dl-space-space-unit);
  margin-bottom: var(--dl-space-space-unit);
  flex-direction: column;
  justify-content: center;
}

.hover-effect:hover {
  transform: scale(1.05);
  /* Adjust scale factor for the desired effect */
  transition: transform 0.3s ease;
  /* Add a smooth transition */
}

.home-container38 {
  width: 100%;
  height: 90%;
  border-color: #E0E0E0;
  border-style: groove;
  border-width: 1px;
}

.home-container39 {
  display: contents;
}

.home-container40 {
  flex: 0 0 auto;
  width: 45%;
  height: 80%;
  display: flex;
  align-self: center;
  align-items: space-between;
  flex-direction: column;
  border-radius: 20px;
  /* Adjust the value to control the roundness of the corners */
  margin-right: 1%;
}

.home-container41 {
  width: 389px;
  height: 100%;
  display: flex;
  align-self: center;
  align-items: center;
  margin-right: var(--dl-space-space-unit);
  flex-direction: column;
  background-size: contain;
  justify-content: flex-start;
  background-image: url("/oie_q3itd7eu4hf6-300h.jpg");
  border-radius: 20px;
  /* Adjust the value to control the roundness of the corners */
}

.home-text12 {
  color: #ffffff;
  font-size: 25px;
  align-self: flex-start;
  margin-top: var(--dl-space-space-halfunit);
  margin-left: var(--dl-space-space-halfunit);
  margin-bottom: var(--dl-space-space-threeunits);
}

.home-text13 {
  color: #ffffff;
  font-size: 25px;
  align-self: flex-start;
  font-style: normal;
  font-weight: 400;
  margin-left: var(--dl-space-space-halfunit);
  margin-bottom: var(--dl-space-space-oneandhalfunits);
}

.home-text14 {
  color: #ffffff;
  font-size: 25px;
  align-self: flex-start;
  margin-left: var(--dl-space-space-halfunit);
  margin-bottom: 56px;
}

.home-container42 {
  width: 100%;
  Height: 10%;
  font-size: 26px;
  align-self: left;
}


.home-container12 {
  width: 100%;
  Height: 100%;
  align-self: left;
}

.home-container43 {
  flex: 0 0 auto;
  width: auto;
  height: 100%;
  display: flex;
  align-items: center;
  margin-left: var(--dl-space-space-unit);
  border-color: #E0E0E0;
  border-style: groove;
  border-width: 1px;
  flex-direction: column;
  justify-content: center;
}

.home-container44 {
  width: 100%;
  height: 20vh;
}

.home-container45 {
  height: 20vh;
}

.home-container46 {
  flex: 0 0 auto;
  width: 200px;
  border: 2px dashed rgba(120, 120, 120, 0.4);
  height: 100%;
  display: flex;
  align-items: flex-start;
  margin-right: var(--dl-space-space-unit);
  flex-direction: column;
}

.home-container47 {
  flex: 0 0 auto;
  width: 200px;
  border: 2px dashed rgba(120, 120, 120, 0.4);
  height: 100%;
  display: flex;
  align-items: flex-start;
  margin-right: var(--dl-space-space-unit);
  flex-direction: column;
}

.home-container48 {
  flex: 0 0 auto;
  width: auto;
  border: 2px dashed rgba(120, 120, 120, 0.4);
  height: auto;
  display: flex;
  align-items: flex-start;
  justify-content: center;
}

@media (max-width: 991px) {
  .home-profile {
    padding-left: var(--dl-space-space-halfunit);
    padding-right: var(--dl-space-space-halfunit);
  }

  .home-container03 {
    margin-left: 0px;
  }
}

@media (max-width: 767px) {
  .home-navbar-interactive {
    padding-left: var(--dl-space-space-twounits);
    padding-right: var(--dl-space-space-twounits);
  }

  .home-desktop-menu {
    display: none;
  }

  .home-burger-menu {
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .home-mobile-menu {
    padding-top: var(--dl-space-space-oneandhalfunits);
    padding-left: var(--dl-space-space-twounits);
    padding-right: var(--dl-space-space-twounits);
    padding-bottom: var(--dl-space-space-oneandhalfunits);
  }

  .home-nav1 {
    margin-bottom: var(--dl-space-space-unit);
  }

  .home-nav2 {
    margin-left: 0;
    margin-bottom: var(--dl-space-space-unit);
  }

  .home-nav3 {
    margin-left: 0;
    margin-bottom: var(--dl-space-space-unit);
  }

  .home-nav4 {
    margin-left: 0;
    margin-bottom: var(--dl-space-space-unit);
  }

  .home-nav5 {
    margin-left: 0;
    margin-bottom: var(--dl-space-space-unit);
  }

  .home-icon12 {
    display: flex;
  }
}

@media (max-width: 479px) {
  .home-navbar-interactive {
    width: 90%;
    padding: var(--dl-space-space-unit);
  }

  .home-logo {
    font-size: 1.5em;
  }

  .home-mobile-menu {
    padding: 16px;
  }

  .home-logo1 {
    font-size: 1.5em;
  }
  .loading-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 50vh; /* Adjust as needed */
  font-size: 1.5em; /* Adjust as needed */
}

}
</style>
