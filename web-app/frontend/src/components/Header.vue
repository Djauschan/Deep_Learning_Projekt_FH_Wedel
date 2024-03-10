<template>
    <div class="Header sticky-top">
        <div class="row align-items-center">
            <div class="d-flex justify-content-end">
                <div v-if="isUserLoggedIn">
                    <DropDownMenu />
                </div>
            </div>
        </div>
    </div>
</template>
  
<script>
import { defineComponent, computed } from 'vue';
import { useStore } from 'pinia';
import DropDownMenu from './DropDownMenu.vue';

export default defineComponent({
    name: 'HeaderPart',
    components: {
        DropDownMenu,
    },
    setup() {
        const store = useStore();

        const isUserLoggedIn = computed(() => {
            return store.state.logged_user !== null;
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
    background-color: red;
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

/* Overwriting Bootstrap's sticky-top styles to make sure it sticks to the top */
.sticky-top {
    position: sticky;
    z-index: 1020;
    /* Ensure it appears above other elements */
}
</style>
  