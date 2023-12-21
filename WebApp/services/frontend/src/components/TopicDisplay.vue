<!-- TopicDisplay.vue -->
<template>
    <div class="topic-container">
      <h1>{{ topic.title }}</h1>
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
    },
    methods: {
      renderHtml(html) {
        return { __html: html };
      },
    },
  };
  </script>
  
  <style scoped>
 .topic-container{
  background-color: #888;
  border: 1px solid #ddd;
  padding: 15px;
  border-radius: 5px;
 }

  h1 {
    color: #333;
  }
  
  p {
    color: #555;
  }
  </style>