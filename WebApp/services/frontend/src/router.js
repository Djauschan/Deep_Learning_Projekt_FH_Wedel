import { createRouter, createWebHistory } from 'vue-router';
import Home from './views/home.vue'
import Statistik from './views/statistik.vue'
import NotFound from './views/not-found.vue'
import Login from './views/Login.vue'
import './style.css'
import { useMyPiniaStore } from './store';

const router = createRouter({
  history: createWebHistory(),
  linkActiveClass: 'active',
  routes: [
    {
      path: '/',
      component: Home,
      meta: { requiresAuth: true }
    },
    {
      path: '/statistik',
      component: Statistik,
      meta: { requiresAuth: true }
    },
    {
      path: '/login',
      component: Login
    },
    {
      path: '/**',
      component: NotFound
    }
  ]
});

router.beforeEach((to) => {
  console.log("loggedId: " + localStorage.isLoggedIn)
  console.log("auth needed: " + to.meta.requiresAuth)
  // Assuming to.meta.requiresAuth is true for routes that require authentication
  if (to.meta.requiresAuth && !localStorage.isLoggedIn) {
    // Redirect to the login page if not authenticated
    console.log("back to login")
    next('/login');
  }
});

export default router;