<template>
    <div class="dropdown-container">
        <button @click="toggleDropdown" class="dropdown-button">☰</button>
        <div v-if="isDropdownOpen" class="overlay">
            <button @click="toggleDropdown" class="dropdown-button2">☰</button>
            <button @click="goToHome" class="menu-item">Home</button>
            <button type="submit" @click="logout" class="menu-item">Logout</button>
        </div>
        <div v-if="isDropdownOpen" class="background" @click="closeDropdown"></div>
    </div>
</template>

<script>
export default {
    components: {
    },
    data() {
        return {
            isDropdownOpen: false
        };
    },
    methods: {
        toggleDropdown() {
            this.isDropdownOpen = !this.isDropdownOpen;
        },
        goToHome() {
            this.closeDropdown();
            this.$router.push('/');
        },
        logout() {
            localStorage.removeItem('logegd_user_id');
            localStorage.removeItem('logged_user');
            this.$store.commit('setUser', null)
            this.$store.commit('setUserId', null)
            this.$router.push('/login')
        },
        closeDropdown() {
            this.isDropdownOpen = false;
        },
    },
};
</script>

<style>
.dropdown-container {
    position: absolute;
    display: inline-block;
    top: 20px;
    left: 10px;
}

.dropdown-button {
    background-color: transparent;
    border: none;
    font-size: 24px;
    cursor: pointer;
    color: purple;
}

.dropdown-button2 {
    background-color: transparent;
    border: none;
    font-size: 24px;
    cursor: pointer;
    position: absolute;
    display: inline-block;
    top: 19px;
    left: 9px;
    color: purple;
}

.menu-item {
    background: none;
    border: none;
    cursor: pointer;
    font-size: 20px;
    padding: 15px 15px;
    width: 100%;
    text-align: left;
    color: purple;
}

.menu-item:hover {
    background-color: #ECF0F1;
}

.background {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    z-index: 999;
    animation: fadeIn 1s;
}

.overlay {
    position: absolute;
    top: -20px;
    left: -10px;
    background-color: #F8F9F9;
    border: 1px solid #ccc;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    padding: 40px;
    font-size: 18px;
    z-index: 1000;
}

/* Animationen */
@keyframes fadeIn {
    from {
        opacity: 0;
    }

    to {
        opacity: 1;
    }
}

@keyframes slideInLeft {
    from {
        transform: translateX(-20%);
    }

    to {
        transform: translateX(0%);
    }
}

.overlay {
    animation: slideInLeft 0.5s;
}
</style>