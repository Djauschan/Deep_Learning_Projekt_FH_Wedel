import { createApp } from 'vue'
import App from './App.vue'
import { createRouter, createWebHashHistory } from "vue-router";
import LoginPage from "./components/Login.vue";
import HomePage from "./components/Homepage";
import store from "./store";

import BootstrapVue3 from "bootstrap-vue-3";

import "bootstrap/dist/css/bootstrap.css";
import "bootstrap-vue-3/dist/bootstrap-vue-3.css";
import "@fortawesome/fontawesome-free/css/all.css";


const app = createApp(App);

createApp(App);

const routes = [
    { path: "/", component: HomePage, meta: { requiresAuth: true } },
    { path: "/login", component: LoginPage },
];

const router = createRouter({
    history: createWebHashHistory(),
    routes: routes,
    linkActiveClass: "active",
});

// always check if user is logged in before allowing routes other then login
router.beforeEach((to, from, next) => {
    if (to.meta.requiresAuth && !store.getters.isLoggedIn) {
        next("/login");
    } else {
        next();
    }
});

// for refresh of page, if user hasn't logged out, the important info gets reloaded into the store
if (
    localStorage.getItem("logged_user") &&
    localStorage.getItem("logged_user_id")
) {
    const user = localStorage.getItem("logged_user");
    const user_id = localStorage.getItem("logged_user_id");
    store.commit("setUser", user);
    store.commit("setUserId", user_id);
}

store.commit("setApi", process.env.VUE_APP_API_URL);
app.use(BootstrapVue3);
app.use(router);
app.use(store);

app.mount("#app");
