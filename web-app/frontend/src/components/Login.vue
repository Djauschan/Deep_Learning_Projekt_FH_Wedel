<template>
    <div class="login-container">
        <div class="animated-background"></div>
        <div class="title-container">Deep Learning Project</div>
        <div class="form-container">
            <h1>{{ showRegister ? "Registrieren" : "Login" }}</h1>
            <form v-if="!showRegister" id="login-form">
                <input type="text" id="username" placeholder="Username" required v-model="username" />
                <input type="password" id="password" placeholder="Password" required v-model="password" />
                <button type="submit" @click.prevent="sendLoginRequest">Log in</button>
                <button type="button" @click.prevent="showRegister = !showRegister">
                    Register
                </button>
            </form>
            <form v-else>
                <input type="text" id="username" placeholder="Username" required v-model="username" />
                <input type="text" id="email" placeholder="Email" required v-model="email" />
                <input type="password" id="password" placeholder="Password" required v-model="password" />
                <input type="password" id="repeat_password" placeholder="Repeat Password" required
                    v-model="repeat_password" />
                <button type="button" @click.prevent="register_new_user">
                    Register
                </button>
                <button type="button" @click.prevent="showRegister = !showRegister">
                    Back to Login
                </button>
            </form>
        </div>
    </div>
</template>

<script>
import axios from "axios";
import { defineComponent, computed } from "vue";
import { useStore } from "pinia";
import Swal from "sweetalert2/dist/sweetalert2.js";
import "sweetalert2/dist/sweetalert2.min.css";
import DropDownMenu from "./DropDownMenu.vue";

export default defineComponent({
    name: "LoginPage",
    components: {
        DropDownMenu,
    },
    setup() {
        const store = useStore();

        const username = ref("");
        const password = ref("");
        const repeat_password = ref("");
        const email = ref("");
        const showRegister = ref(false);
        const userLocation = reactive({ lat: null, lng: null });

        const isUserLoggedIn = computed(() => {
            return store.state.logged_user !== null;
        });

        const register_new_user = async () => {
            if (showRegister.value) {
                if (repeat_password.value === password.value) {
                    try {
                        const response = await axios.post(store.state.API + "/createUser", {
                            email: email.value,
                            username: username.value,
                            password: password.value,
                        });
                        console.log(response.status);
                        showSuccess();
                    } catch (error) {
                        console.log(error);
                        showError("Fehler beim Registrieren", "Username bereits belegt");
                    }
                } else {
                    showError(
                        "Fehler beim Registrieren",
                        "Bitte versuche es noch einmal und gib zweimal das gleiche Passwort ein :)"
                    );
                }
            }
        };

        const sendLoginRequest = async () => {
            try {
                const response = await axios.post(store.state.API + "/login/", {
                    user: username.value,
                    password: password.value,
                    location: userLocation,
                });
                if (response.data) {
                    getUser(response.data.username);
                    store.login({
                        user: response.data.username,
                        user_id: response.data.id,
                    });
                    localStorage.setItem("logged_user", response.data.username);
                    localStorage.setItem("logged_user_id", response.data.id);
                    router.push("/");
                }
            } catch (error) {
                Swal.fire({
                    title: "Fehler beim Login",
                    text: "Falsche Logindaten eingegeben",
                    icon: "info",
                    iconColor: "#d0342c",
                    showCloseButton: false,
                    confirmButtonText: "Schließen",
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
        };

        const getUser = (name) => {
            username.value = name;
        };

        const showSuccess = () => {
            Swal.fire({
                title: "Erfolgreich registriert",
                text: "Du kannst dich jetzt einloggen",
                icon: "info",
                iconColor: "#2200cd",
                showCloseButton: false,
                confirmButtonText: "Schließen",
                confirmButtonColor: "#2200cd",
            }).then((result) => {
                if (result.value) {
                    console.log("Hi");
                    this.showRegister = !this.showRegister;
                } else {
                    console.log("ciao");
                }
            });
        };

        const showError = (title, text) => {
            Swal.fire({
                title: title,
                text: text,
                icon: "info",
                iconColor: "#d0342c",
                showCloseButton: false,
                confirmButtonText: "Schließen",
                confirmButtonColor: "#d0342c",
            });
        };

        return {
            username,
            password,
            repeat_password,
            email,
            showRegister,
            userLocation,
            isUserLoggedIn,
            register_new_user,
            login_try,
            sendLoginRequest,
            getUser,
            showSuccess,
            showError,
        };
    },
});
</script>
  
<script>
import axios from "axios";
import { mapState } from "vuex";
import Swal from "sweetalert2/dist/sweetalert2.js";
import "sweetalert2/dist/sweetalert2.min.css";
export default {
    name: "LoginPage",
    data() {
        return {
            username: "",
            password: "",
            repeat_password: "",
            email: "",
            API: this.$store.state.API,
            showRegister: false,
            userLocation: { lat: null, lng: null },
        };
    },
    methods: {
        // method to call api route to create a new user in the database
        async register_new_user() {
            if (this.showRegister) {
                if (this.repeat_password === this.password) {
                    try {
                        const response = await axios.post(
                            this.$store.state.API + "/createUser",
                            {
                                email: this.email,
                                username: this.username,
                                password: this.password,
                            }
                        );
                        console.log(response.status);
                        this.showSuccess();
                    } catch (error) {
                        console.log(error)

                        Swal.fire({
                            title: "Fehler beim Registrieren",
                            text: "Username bereits belegt",
                            icon: "info",
                            iconColor: "#d0342c",
                            showCloseButton: false,
                            confirmButtonText: "Schließen",
                            confirmButtonColor: "#d0342c",
                        });
                    }
                } else {
                    Swal.fire({
                        title: "Fehler beim Registrieren",
                        text: "Bitte versuche es noch einmal und gib zweimal das gleiche Passwort ein :)",
                        icon: "info",
                        iconColor: "#d0342c",
                        showCloseButton: false,
                        confirmButtonText: "Schließen",
                        confirmButtonColor: "#d0342c",
                    });
                }
            }
        },

        // method to create a Login for a user that exists in the db, also sets items in the vuex state management and localstorage
        async sendLoginRequest() {
            try {
                const response = await axios.post(this.$store.state.API + "/login/", {
                    user: this.username,
                    password: this.password,
                    location: this.userLocation,
                });
                if (response.data) {
                    this.getUser(response.data.username);
                    this.$store.dispatch("login", {
                        user: response.data.username,
                        user_id: response.data.id,
                    });
                    localStorage.setItem("logged_user", response.data.username);
                    localStorage.setItem("logged_user_id", response.data.id);
                    this.$router.push("/");
                }
            } catch (error) {
                Swal.fire({
                    title: "Fehler beim Login",
                    text: "Falsche Logindaten eingegeben",
                    icon: "info",
                    iconColor: "#d0342c",
                    showCloseButton: false,
                    confirmButtonText: "Schließen",
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
        getUser(name) {
            this.username = name;
        },
    },
    computed: {
        ...mapState(["user"]),
        isLoggedIn() {
            return this.$store.getters.isLoggedIn;
        },
        currentUser() {
            return this.$store.getters.currentUser;
        },
    },
};
</script>
  
<style>
.login-container {
    position: relative;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    overflow: hidden;
}

.animated-background {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(180deg, #142957, white);
    background-size: 300% 300%;
    animation: gradientAnimation 15s ease infinite;
}

@keyframes gradientAnimation {
    0% {
        background-position: 0% 0%;
    }

    50% {
        background-position: 100% 50%;
    }

    100% {
        background-position: 0% 0%;
    }
}

.title-container {
    z-index: 1;
    text-align: left;
    max-width: 600px;
    height: 40%;
    padding-left: 10%;
    font-size: 60px;
    color: white;
    font-family: "Courier New", monospace;
}

.form-container {
    z-index: 1;
    padding: 20px;
    background-color: rgba(255, 255, 255, 0.9);
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
    border-radius: 10px;
    text-align: center;
    width: 35vh;
    margin: 0 auto;
    font-family: "Trebuchet MS", sans-serif;
}

.form-container h1 {
    font-size: 24px;
    margin-bottom: 20px;
    color: #142957;
}

.form-container input[type="text"],
.form-container input[type="password"],
.form-container input[type="email"] {
    width: 100%;
    padding: 12px;
    margin-bottom: 20px;
    border: 1px solid #ccc;
    border-radius: 6px;
    box-sizing: border-box;
    font-size: 16px;
}

.form-container button {
    width: 100%;
    padding: 12px;
    background-color: #2200cd;
    color: #fff;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 18px;
}

.form-container button:hover {
    background-color: #17008a;
}

.form-container form {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.form-container button[type="button"] {
    background-color: #17008a;
    margin-top: 10px;
}

.form-container button[type="button"]:hover {
    background-color: #142957;
}
</style>
  