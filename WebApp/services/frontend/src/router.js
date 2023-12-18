import { createRouter, createWebHistory } from "vue-router";
import Home from "./views/home.vue";
import Statistik from "./views/statistik.vue";
import NotFound from "./views/not-found.vue";
import Login from "./views/Login.vue";
import "./style.css";

const router = createRouter({
  history: createWebHistory(),
  linkActiveClass: "active",
  routes: [
    {
      path: "/",
      component: Home,
      meta: { requiresAuth: false },
    },
    {
      path: "/statistik",
      component: Statistik,
      meta: { requiresAuth: false },
    },
    {
      path: "/login",
      component: Login,
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
