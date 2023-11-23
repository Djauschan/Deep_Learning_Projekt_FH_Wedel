
<template>
  <router-view/>
</template>

<script>
export default {
  name: "App",
  components: {
    HeaderPart,
  },
  computed: {
    ...mapState(['logged_user']),
    isUserLoggedIn() {
      return this.logged_user !== null;
    },
  },
};
</script>