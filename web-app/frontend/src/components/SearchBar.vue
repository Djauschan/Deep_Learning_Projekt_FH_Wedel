<!-- SearchBar.vue -->
<template>
    <div>
      <input
        v-model="searchQuery"
        ref="searchbar"
        @input="handleInput"
        placeholder="Search..."
        class="short-search-bar"
      >
      <button @click="handleSearch">Search</button>
  
      <!-- Display live results with a limit -->
      <ul>
        <li v-for="(result, index) in filteredResults" :key="result.id">
          {{ result.name }}
        </li>
      </ul>
    </div>
  </template>
  
  <script>
  export default {
    data() {
      return {
        searchQuery: '',
        results: [
          { id: 1, name: 'E-Mart' },
          { id: 2, name: 'Amazon' },
          // Add sample results
        ],
        maxResults: 1, // Set the maximum number of previewed results
      };
    },
    computed: {
      filteredResults() {
        const filtered = this.results.filter(result =>
          result.name.toLowerCase().includes(this.searchQuery.toLowerCase())
        );
        return filtered.slice(0, this.maxResults);
      },
    },
    methods: {
      handleInput() {
        //perform any other real-time actions here
        console.log('Input:', this.searchQuery);
      },
      handleSearch() {
        //trigger the search action here
        console.log('Search:', this.searchQuery);
      },
    },
  };
  </script>
  
  <style scoped>
  .short-search-bar {
    height: 30px;
    width: 120px;
  }
  </style>