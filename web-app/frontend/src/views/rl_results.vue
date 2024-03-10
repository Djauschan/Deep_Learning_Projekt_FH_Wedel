<template>
  <div>
    <Header />
    <div v-if="rlData">
      <pre>{{ rlData }}</pre>
    </div>
  </div>
</template>

<script>
import { useMyPiniaStore } from "../store.js"; // Import your store
import axios from "axios";
import Header from './Header.vue'
export default {
  name: 'RlResults',
  components: {
    Header
  },
  data() {
    return {
      rlData: null,
      store: useMyPiniaStore() // Use your store
    };
  },
  methods: {
    async load_rl_data() {
      try {
        const response = await axios.get(`${this.store.API}/predict/rl`, { // Use store.state.API
          params: {
            stock_symbols: '[AAPL, AAL, SNAP, TSLA]',
            start_date: '2021-01-04',
            end_date: '2021-01-08',
            resolution: 'H'
          }
        });
        this.rlData = response.data;
      } catch (error) {
        console.error(error);
      }
    }
  },
  created() {
    this.load_rl_data();
  }
}
</script>

<style scoped>
/* Your styles here */
</style>