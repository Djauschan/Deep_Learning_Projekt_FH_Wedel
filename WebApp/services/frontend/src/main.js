import Vue from "vue";
import App from "./App.vue";
import router from "./router";
import VueRouter from "vue-router";
import LottieVuePlayer from "@lottiefiles/vue-lottie-player";

Vue.use(VueRouter);

/*
const router = new VueRouter({
  routes,
});
*/

Vue.use(LottieVuePlayer);
Vue.config.productionTip = false;

new Vue({
  render: (h) => h(App),
  router,
}).$mount("#app");
