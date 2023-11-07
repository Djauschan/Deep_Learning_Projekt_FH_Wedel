<template>
    <div class="test">
        <div>{{ username }}</div>
    </div>
</template>

<script>

import axios from "axios";

export default {
    name: "HomePage",
    data() {
        return {
            user: this.$store.state.logged_user_id,
            username: this.$store.state.logged_user,
            API: this.$store.state.API,
        };
    },
    methods: {
        // method to get initial Userdata
        async fetchUserData() {
            try {
                const response = await axios.get(
                    this.$store.state.API + `/users/${this.$store.state.logged_user_id}`
                );
                this.userObject = response.data;
            } catch (error) {
                console.log(error);
            }
        },
    },
};
</script>
  
<style scoped>
.row {
    background-color: #3c4e74;
}


.title {
    font-size: 24px;
    margin-bottom: 10px;
}

.info-item {
    margin-bottom: 5px;
}
</style>
  