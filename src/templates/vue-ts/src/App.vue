<template>
  <div id="app">
    <header>
      <nav>
        <RouterLink to="/">Home</RouterLink>
        <RouterLink to="/about">About</RouterLink>
      </nav>
    </header>

    <main>
      <RouterView />
    </main>
  </div>
</template>

<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
</script>

<style scoped>
header {
  background-color: #2c3e50;
  padding: 1rem;
  margin-bottom: 2rem;
}

nav {
  display: flex;
  gap: 1rem;
}

nav a {
  color: #ecf0f1;
  text-decoration: none;
  font-weight: 500;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  transition: background-color 0.3s;
}

nav a:hover,
nav a.router-link-active {
  background-color: #3498db;
}

main {
  padding: 0 2rem;
}
</style> 