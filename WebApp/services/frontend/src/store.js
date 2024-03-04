import { defineStore } from 'pinia';

// Initial state function
const initialState = () => ({
    logged_user: null,
    logged_budget: 0,
    logged_user_id: null,
    API: 'http://localhost:8000'
});

export const useMyPiniaStore = defineStore({
    id: 'myPiniaStore',
    state: initialState,
    mutations: {
        setUser(state, user) {
            state.logged_user = user;
        },
        setBudget(state, budget) {
            state.logged_budget = budget;
            localStorage.setItem('budget', budget);
        },
        setUserId(state, user_id) {
            state.logged_user_id = user_id;
        },
        setApi(state, API) {
            state.API = API;
        },
        RESET_STATE(state) {
            Object.assign(state, initialState());
        },
    },
    getters: {
        currentUser: (state) => state.logged_user,
        getBudget: (state) => state.logged_budget,
        isLoggedIn: (state) => state.logged_user !== null,
    },
    actions: {
        login({ commit }, { user, user_id }) {
            commit("setUser", user);
            commit("setUserId", user_id);
        },
        logout({ commit }) {
            commit("setUser", null);
            commit("RESET_STATE");
        },
        resetState({ commit }) {
            commit("RESET_STATE");
        },
    }
});
