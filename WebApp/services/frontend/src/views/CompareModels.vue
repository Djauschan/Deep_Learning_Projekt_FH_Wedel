<template>
  <button class="button-stock" @click="test">One Click Solution</button>
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