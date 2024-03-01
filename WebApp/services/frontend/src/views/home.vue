<template>
  <Header />
  <div :class="{ 'dark-mode': isDarkMode }">
  <div class="home-container">
    <div class="home-header">
      <header data-thq="thq-navbar" class="home-navbar-interactive">
        <div class="home-container01">
          <span class="home-logo">Alles auf einen Blick!</span>
          <button ref="myButton1" @mouseover="changeCursor" @mouseleave="resetCursor"  @click="toggleDarkMode">Toggle Dark Mode</button>
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
                <div class="home-container30">
                  <div class="home-container31">
                    <span>Aktie</span>
                  </div>
                  <div class="spread">
                    <div class="home-container32">
                      <svg viewBox="0 0 1024 1024" class="home-icon30">
                        <path
                          d="M914.005 409.344l-275.84-278.827c-56.704-56.747-131.712-87.851-211.499-87.851s-154.795 31.104-211.2 87.509c-60.416 60.331-90.752 143.189-86.485 226.731-28.245 46.293-43.648 99.413-43.648 155.051 0 79.787 31.104 154.795 87.509 211.2l130.261 127.317 148.565 148.48c16.683 16.725 38.485 25.045 60.331 25.045s43.648-8.32 60.331-25.003l298.667-298.667c33.195-33.195 33.365-86.955 0.341-120.32l-3.968-4.011 46.293-46.336c33.195-33.195 33.365-86.955 0.341-120.32zM512 938.667l-149.248-149.205-129.579-126.635c-83.328-83.328-83.328-218.411 0-301.653 41.643-41.685 96.256-62.507 150.827-62.507s109.184 20.821 150.827 62.507l275.84 278.827-298.667 298.667zM595.499 301.184c-56.704-56.747-131.712-87.851-211.499-87.851-54.443 0-106.539 14.805-152.107 41.899 10.283-23.467 24.704-45.525 43.947-64.768 41.643-41.643 96.256-62.464 150.827-62.464s109.184 20.821 150.827 62.507l275.84 278.827-45.995 45.995-211.84-214.144z">
                        </path>
                        <path
                          d="M384 447.957c35.413 0 64 28.672 64 64.043 0 35.413-28.587 63.957-64 63.957s-64-28.544-64-63.957c0-35.371 28.587-64.043 64-64.043M384 405.291c-58.795 0-106.667 47.872-106.667 106.709 0 58.795 47.872 106.624 106.667 106.624s106.667-47.829 106.667-106.624c0-58.837-47.872-106.709-106.667-106.709z">
                        </path>
                      </svg>
                      <div class="buttonBG">
                      <select ref="myButton4" @mouseover="changeCursor" @mouseleave="resetCursor">
                        <option value="Option 1" selected>
                          E-Mart
                        </option>
                        <option value="Option 2">
                          Amazon
                        </option>
                      </select>
                      </div>
                      <div class="search">
                        <search-bar ref="searchbar"></search-bar>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              </div>
              <div class="seperator"></div>
              <div class="home-container25">
                <div class="hover-effect">
                <div class="home-container26">
                  <div class="home-container27">
                    <span>aktives Modell</span>
                  </div>
                  <div class="home-container28">
                    <svg viewBox="0 0 1024 1024" class="home-icon26">
                      <path
                        d="M512 682.667h-341.333c-11.776 0-22.4-4.736-30.165-12.501s-12.501-18.389-12.501-30.165v-426.667c0-11.776 4.736-22.4 12.501-30.165s18.389-12.501 30.165-12.501h682.667c11.776 0 22.4 4.736 30.165 12.501s12.501 18.389 12.501 30.165v426.667c0 11.776-4.736 22.4-12.501 30.165s-18.389 12.501-30.165 12.501zM469.333 768v85.333h-128c-23.552 0-42.667 19.115-42.667 42.667s19.115 42.667 42.667 42.667h341.333c23.552 0 42.667-19.115 42.667-42.667s-19.115-42.667-42.667-42.667h-128v-85.333h298.667c35.328 0 67.413-14.379 90.496-37.504s37.504-55.168 37.504-90.496v-426.667c0-35.328-14.379-67.413-37.504-90.496s-55.168-37.504-90.496-37.504h-682.667c-35.328 0-67.413 14.379-90.496 37.504s-37.504 55.168-37.504 90.496v426.667c0 35.328 14.379 67.413 37.504 90.496s55.168 37.504 90.496 37.504z">
                      </path>
                    </svg>
                    <div class="buttonBG">
                    <select ref="myButton5" @mouseover="changeCursor" @mouseleave="resetCursor">
                      <option value="Option 1" selected>CNN</option>
                      <option value="Option 2">ANN</option>
                      <option value="Option 1">Transformer</option>
                      <option value="Option 2">LSTM</option>
                    </select>
                    </div>
                  </div>
                </div>
              </div>
              </div>
              <div class="seperator"></div>
              <div class="home-container29">
                <div class="hover-effect">
                <div class="home-container26">
                  <div class="home-container23">
                    <span>Modell Guthaben</span>
                  </div>
                  <div class="home-container24">
                    <svg viewBox="0 0 1024 1024" class="home-icon28">
                    <path
                      d="M928 128h-832c-52.8 0-96 43.2-96 96v576c0 52.8 43.2 96 96 96h832c52.8 0 96-43.2 96-96v-576c0-52.8-43.2-96-96-96zM96 192h832c17.346 0 32 14.654 32 32v96h-896v-96c0-17.346 14.654-32 32-32zM928 832h-832c-17.346 0-32-14.654-32-32v-288h896v288c0 17.346-14.654 32-32 32zM128 640h64v128h-64zM256 640h64v128h-64zM384 640h64v128h-64z">
                    </path>
                    </svg>
                    <input type="text" placeholder="Wert in Euro" class="home-textinput input" />
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
                  <button ref="myButton2" @mouseover="changeCursor" @mouseleave="resetCursor"  type="button" class="button" @click="simulateProgress">Prognose</button>
                </div>
              </div>
              </div>
              <div class="seperator"></div>
              <div class="hover-effect">
              <div class="home-container70">
               <div class="home-container33">
                <div class="home-container34">
                  <span>Der Agent empfiehlt:</span>
                </div>
                <button ref="myButton" @mouseover="changeCursor" @mouseleave="resetCursor" type="button" class="button">Aktie Kaufen!</button>
               </div>
              </div>
              </div>
            <div class="seperator"></div>
            <div class="hover-effect">
            <div class="home-container71">
              <div class="home-container33">
                <div class="home-container34">
                  <span>Zuletzt angesehen</span>
                </div>
                <div class="home-container35">
                 <select ref="myButton3" @mouseover="changeCursor" @mouseleave="resetCursor">
                   <option value="Option 1" selected>
                     E-Mart
                   </option>
                   <option value="Option 2">
                    Amazon
                   </option>
                  </select>
                </div>
              </div>
            </div>
            </div>
            </div>
          </div>
          <div class="home-container36">
            <div class="home-container37">
              <div class="home-container38">
                <div class="home-container39">
                  <DxChart id="chart" :data-source="dataSource" title="Predicted Stock Price">
                   <DxCommonSeriesSettings argument-field="date" type="stock" />
                    <DxSeries name="E-Mart" open-value-field="o" high-value-field="h" low-value-field="l" close-value-field="c">
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
                </div>
              </div>
            </div>
            <div class="home-container40">
              <div class="home-container43">
                <div class="home-container42">
                 <span>Additional Information</span>
                </div>
                <span>Model Calculating...</span>
                <div class="home-container12">
                 <ProgressBar ref="progressBar" />
                </div>
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

<script>import DxChart, {
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
  data() {
    return {
      dataSource,
      isDarkMode: false,
    };
  },
  methods: {
    changeCursor() {
      this.$refs.myButton.style.cursor = 'pointer'; // or any other cursor value
      this.$refs.myButton1.style.cursor = 'pointer'; // or any other cursor value
      this.$refs.myButton2.style.cursor = 'pointer'; // or any other cursor value
      this.$refs.myButton3.style.cursor = 'pointer'; // or any other cursor value
      this.$refs.myButton4.style.cursor = 'pointer'; // or any other cursor value
      this.$refs.myButton5.style.cursor = 'pointer'; // or any other cursor value
    },
    resetCursor() {
      this.$refs.myButton.style.cursor = 'auto'; // reset to the default cursor
      this.$refs.myButton1.style.cursor = 'auto'; // reset to the default cursor
      this.$refs.myButton2.style.cursor = 'auto'; // reset to the default cursor
      this.$refs.myButton3.style.cursor = 'auto'; // reset to the default cursor
      this.$refs.myButton4.style.cursor = 'auto'; // reset to the default cursor
      this.$refs.myButton5.style.cursor = 'auto'; // reset to the default cursor
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
  height: 100vh; /* Set initial height to 100% of viewport height */
  flex:1;
}

.dark-mode {
    background-color: #333; /* Dark background color */
    color: #fff; /* Light text color for better contrast */
  }



.loading-spinner {
    border: 4px solid #f3f3f3; /* Light gray border */
    border-top: 4px solid #3498db; /* Blue border for spinner animation */
    border-radius: 50%;
    width: 20px;
    height: 20px;
    animation: spin 1s linear infinite; /* Spinner animation */
  }

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }

#chart {
  height: 100%;
}

.search {
  width: 100%;
  object-fit: cover;
  margin-left: 15%;
}

.spread{
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
  font-size: 75px;
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
  border-color:  #E0E0E0;
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
  align-items: center; /* Center items along the cross axis */
  justify-content: space-between; /* Distribute items along the main axis with equal space */
}

.seperator{
  border: 1px solid #E0E0E0;
  height: 100%;
}

.home-container22 {
  flex: 0 0 auto;
  width: 320px;
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
  border-radius: 10px; /* Adjust the value to control the roundness of the corners */
}

.home-container24 {
  flex: 0 0 auto;
  width: 100%;
  height: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 10px; /* Adjust the value to control the roundness of the corners */
}

.home-icon26 {
  width: 24px;
  height: 24px;
  margin-right: var(--dl-space-space-halfunit);
  border-radius: 10px; /* Adjust the value to control the roundness of the corners */
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

.home-container27 {
  width: 200px;
  height: 27px;
  display: flex;
  align-items: center;
  flex-direction: column;
  justify-content: center;
  border-radius: 10px; /* Adjust the value to control the roundness of the corners */
}

.home-container28 {
  flex: 0 0 auto;
  width: 266px;
  height: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 10px; /* Adjust the value to control the roundness of the corners */
}

.home-icon28 {
  width: 24px;
  height: 24px;
  margin-right: var(--dl-space-space-halfunit);
  border-radius: 10px; /* Adjust the value to control the roundness of the corners */
}

.home-textinput {
  width: 174px;
  height: 24px;
  border-radius: 10px; /* Adjust the value to control the roundness of the corners */
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
  border-radius: 10px; /* Adjust the value to control the roundness of the corners */
}

.home-container32 {
  flex: 0 0 auto;
  width: 100%;
  height: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 10px; /* Adjust the value to control the roundness of the corners */
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
  border-radius: 10px; /* Adjust the value to control the roundness of the corners */
}

.home-container33 {
  flex: 0 0 auto;
  width: 200px;
  height: 100px;
  display: flex;
  align-items: center;
  margin-right: var(--dl-space-space-unit);
  flex-direction: column;
  justify-content: center;
}

.home-container34 {
  width: 200px;
  height: 27px;
  display: flex;
  align-items: center;
  flex-direction: column;
  justify-content: center;
  border-radius: 10px; /* Adjust the value to control the roundness of the corners */
}

.home-container35 {
  flex: 0 0 auto;
  width: 266px;
  height: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 10px; /* Adjust the value to control the roundness of the corners */
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
  width: 70%;
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
    transform: scale(1.05); /* Adjust scale factor for the desired effect */
    transition: transform 0.3s ease; /* Add a smooth transition */
  }

.home-container38 {
  width: 100%;
  height: 90%;
  border-color:  #E0E0E0;
  border-style: groove;
  border-width: 1px;
}

.home-container39 {
  display: contents;
}

.home-container40 {
  flex: 0 0 auto;
  width: 25%;
  height: 60%;
  display: flex;
  align-self: center;
  align-items: space-between;
  flex-direction: column;
  border-radius: 20px; /* Adjust the value to control the roundness of the corners */
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
  border-radius: 20px; /* Adjust the value to control the roundness of the corners */
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
  Height: 100%;
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
  border-color:  #E0E0E0;
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

}
</style>
