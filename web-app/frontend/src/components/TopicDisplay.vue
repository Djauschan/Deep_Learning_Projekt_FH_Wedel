<template>
  <div class="topic-container">
    <h1>{{ topic.title }}</h1>
    <img :src="imageUrl" alt="Topic Image" v-if="imageUrl" :style="{ maxWidth: imageMaxWidth, maxHeight: imageMaxHeight }" />
    <p v-html="topic.description"></p>
  </div>
</template>

<script>
export default {
  props: {
    topic: {
      type: Object,
      required: true,
    },
    imageUrl: {
      type: String,
      default: '',
    },
    imageMaxWidth: {
      type: String,
      default: '100%', // Default max-width
    },
    imageMaxHeight: {
      type: String,
      default: 'auto', // Default max-height
    },
  },
};
</script>

<style scoped>
.topic-container {
  background-color: #3498db; /* Updated background color */
  border: 1px solid #ddd;
  padding: 15px;
  border-radius: 10px; /* Increased border radius */
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Added box shadow */
}

h1 {
  color: #fff; /* Changed text color to white */
  font-size: 24px; /* Increased font size */
  margin-bottom: 10px; /* Added margin bottom */
}

p {
  color: #333; /* Changed text color to a darker shade */
  line-height: 1.5; /* Increased line height for better readability */
}
</style>
