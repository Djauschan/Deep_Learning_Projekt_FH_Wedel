<template>
    <div class="Header sticky-top">
        <div class="row">
            <div class="col-md-0">
                <div class="logo-container">Deep Learning Project</div>
            </div>
        </div>
    </div>
</template>
  
<script>
export default {
    name: "HeaderPart",
    data() {
        return {};
    },
    methods: {},
};
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

/* Overwriting Bootstrap's sticky-top styles to make sure it sticks to the top */
.sticky-top {
    position: sticky;
    z-index: 1020;
    /* Ensure it appears above other elements */
}
</style>
  