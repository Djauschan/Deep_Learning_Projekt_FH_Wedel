<template>
  <div>
    <div v-if="isUserLoggedIn">
      <HeaderPart class="header"></HeaderPart>
    </div>
    <router-view></router-view>
  </div>
</template>

<script>
import HeaderPart from "./components/Header.vue";
import { mapState } from 'vuex';
export default {
  name: "App",
  components: {
    HeaderPart,
  },
  computed: {
    ...mapState(['logged_user']),
    isUserLoggedIn() {
      return this.logged_user !== null;
    },
  },
};
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  background-color: #f8f9f9;
}
</style>