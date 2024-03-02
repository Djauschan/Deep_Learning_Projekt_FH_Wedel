import { createRouter, createWebHistory } from "vue-router";
import Home from "./views/home.vue";
import Statistik from "./views/statistik.vue";
import ModelInformation from "./views/ModelInformation.vue";
import StockList from "./views/StockList.vue";
import NotFound from "./views/not-found.vue";
import Login from "./views/Login.vue";
import rl_results from "./views/rl_results.vue";
import "./style.css";

const router = createRouter({
  history: createWebHistory(),
  linkActiveClass: "active",
  routes: [
    {
      path: "/",
      component: Home,
      meta: { requiresAuth: true },
    },
    {
      path: "/statistik",
      component: Statistik,
      meta: { requiresAuth: true },
    },
    {
      path: "/ModelInformation",
      component: ModelInformation,
      meta: { requiresAuth: true },
    },
    {
      path: "/StockList",
      component: StockList,
      meta: { requiresAuth: true },
    },
    {
      path: "/login",
      component: Login,
    },
    {
      path: "/rl_results",
      component: rl_results,
    },
    {
      path: "/**",
      component: NotFound,
    },
  ],
});

router.beforeEach((to, from, next) => {
  console.log("loggedId: " + localStorage.getItem("isLoggedIn"));
  console.log("auth needed: " + to.meta.requiresAuth);
  if (to.meta.requiresAuth && localStorage.getItem("isLoggedIn") !== "true") {
    console.log("back to login");
    next("/login");
  } else {
    console.log("not back to login");
    next();
  }
});

export default router;
