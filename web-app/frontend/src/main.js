import { createApp } from 'vue';
import App from './App.vue';
import router from './router';
import { createPinia } from 'pinia';

const app = createApp(App);
const pinia = createPinia();

// Use Pinia as a plugin
app.use(pinia)

// Use the router with the app
app.use(router);

// Mount the app
app.mount('#app');
