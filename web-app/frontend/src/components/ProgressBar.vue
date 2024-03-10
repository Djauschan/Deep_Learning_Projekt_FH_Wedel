<template>
  <div class="progress-bar-container">
    <div :style="{ width: progress + '%' }" class="progress-bar"></div>
    <div class="percentage-text">{{ progress }}%</div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      progress: 0,
    };
  },
  methods: {
    updateProgress(value) {
      this.progress = value;
    },
  },
};
</script>

<style scoped>
.progress-bar-container {
  position: relative;
  width: 100%;
  height: 20px;
  background-color: #ddd;
  margin-bottom: 10px;
}

.progress-bar {
  position: absolute;
  top: 0;
  left: 0;
  height: 100%;
  background-color: #4caf50;
  transition: width 0.3s;
}

.percentage-text {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  color: #000000;
}
</style>