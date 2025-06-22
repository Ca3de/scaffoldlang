<script lang="ts">
  import { onMount } from 'svelte';
  
  let count = 0;
  let mounted = false;

  onMount(() => {
    mounted = true;
  });

  const increment = () => {
    count += 1;
  };
</script>

<svelte:head>
  <title>{{projectName}}</title>
  <meta name="description" content="A modern SvelteKit application" />
</svelte:head>

<main class="flex min-h-screen flex-col items-center justify-center p-8">
  <div class="text-center max-w-4xl">
    <h1 class="text-6xl font-bold mb-8">
      Welcome to <span class="text-orange-500">{{projectName}}</span>
    </h1>
    
    <p class="text-xl text-gray-600 mb-12 max-w-2xl mx-auto">
      A modern SvelteKit application built with TypeScript and Tailwind CSS.
      {#if mounted}Svelte is ready!{/if}
    </p>

    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
      <div class="p-6 border border-gray-200 rounded-lg hover:shadow-md transition-shadow">
        <h3 class="text-xl font-semibold mb-2">🚀 SvelteKit</h3>
        <p class="text-gray-600">Modern full-stack framework with excellent DX</p>
      </div>
      
      <div class="p-6 border border-gray-200 rounded-lg hover:shadow-md transition-shadow">
        <h3 class="text-xl font-semibold mb-2">🔧 TypeScript</h3>
        <p class="text-gray-600">Full type safety with Svelte's compiler magic</p>
      </div>
      
      <div class="p-6 border border-gray-200 rounded-lg hover:shadow-md transition-shadow">
        <h3 class="text-xl font-semibold mb-2">🎨 Tailwind CSS</h3>
        <p class="text-gray-600">Utility-first CSS with excellent Svelte integration</p>
      </div>
    </div>

    <div class="flex gap-4 justify-center items-center">
      <button
        on:click={increment}
        class="bg-orange-500 hover:bg-orange-600 text-white px-6 py-3 rounded-lg transition-colors"
      >
        Count: {count}
      </button>
      
      <a
        href="/about"
        class="border border-gray-300 hover:border-gray-400 px-6 py-3 rounded-lg transition-colors"
      >
        About Page
      </a>
    </div>
  </div>
</main>

<style>
  /* Custom styles can go here */
</style> 