<template>
    <div class="Header sticky-top">
        <div class="row align-items-center">
            <div class="d-flex justify-content-end">
                <div v-if="isUserLoggedIn">
                    <DropDownMenu />  
                    <div class="extendedHeader">
                      <img alt="image" src="/external/fh_wedel_logo.svg-200h.png" class="home-image2" />
                      <div class="LoginPicText">
                       <div class="LoginPic">
                          <img alt="image" src="/profil-200h.jpg" class="home-image1" />
                       </div>
                       <div class="LoginText">
                        <span class="home-text">Thomas <br></span>
                        <span class="home-text01">View Profile</span>
                       </div>
                      </div>
                   </div>
                </div>
            </div>
        </div>
    </div>
</template>
  
<script>
import { defineComponent, computed } from 'vue';
import DropDownMenu from './DropDownMenu.vue';

export default defineComponent({
    name: 'HeaderPart',
    components: {
        DropDownMenu,
    },
    setup() {

        const isUserLoggedIn = computed(() => {
            return localStorage.logged_user !== null;
        });

        return {
            isUserLoggedIn,
        };
    },
});
</script>
  
<style>
.row {
    max-width: 100%;
}

.Header {
    background-color: #3c4e74;
    height: 100px;
    border-bottom: 1px solid grey;
}

.logo-container {
    position: absolute;
    top: 30px;
    text-align: center;
    width: 100%;
    height: 110px;
    font-size: 24px;
    font-family: "Courier New", monospace;
    color: #fff;
}

.sticky-top {
    position: sticky;
    z-index: 1020;
    /* Ensure it appears above other elements */
}

.home-image2 {
  height: 100px;
  object-fit: cover; 
  margin-left: 2%;
}

.home-image1 {
  border-radius: 20px; /* Adjust the value to control the roundness of the corners */
}

.LoginPicText {
  background-color: white; /* Set the background color to white */
  width: 18vh;
  height: 8vh;
  object-fit: cover;
  margin-top: 10px;
  margin-left: auto;
  margin-right: 5%;
  display: flex;
  justify-content: space-between;
  border-color: #7e7e7e;
  border-width: 1px;
  padding-top: 0.5%;
  padding-left: 0.5%;
  padding-right: 0.5%;
  padding-bottom: 0.5%;
  overflow: hidden; /* Clear the float */
  border-radius: 10px; /* Adjust the value to control the roundness of the corners */
}

.LoginPic {
    padding-top: 10px;
    margin-right: 10px; /* Add some margin for spacing */
}

.LoginText {
    padding-top: 10px;
    overflow: hidden; /* Clear the float */
}

.extendedHeader {
  display: flex;
  justify-content: space-between;
}
</style>