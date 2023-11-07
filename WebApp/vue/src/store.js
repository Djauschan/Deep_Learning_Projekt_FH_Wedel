import { createStore } from "vuex";

// Initial state function
const initialState = () => ({
    logged_user: null,
    logged_user_id: null,
});

export default createStore({
    state: initialState(),
    mutations: {
        setUser(state, user) {
            state.logged_user = user;
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
    actions: {
        login({ commit }, { user, user_id }) {
            commit("setUser", user);
            commit("setUserId", user_id);
        },
        logout({ commit }) {
            commit("setUser", null);
            commit("RESET_STATE");
        },
        // action to execute the reset
        resetState({ commit }) {
            commit("RESET_STATE");
        },
    },
    getters: {
        currentUser: (state) => state.logged_user,
        isLoggedIn: (state) => state.logged_user !== null,
    },
});
