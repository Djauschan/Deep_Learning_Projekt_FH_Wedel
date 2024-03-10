<template>
  <div>
    <Header />
    <div class="background">
      <div class="subhead">
        <div class="uberschrift">
          <span class="home-logo">Informationen zu Stocks</span>
        </div>
      </div>
      <input type="text" v-model="searchTerm" placeholder="Search topics">
      <div class="topics-container">
       <TopicDisplay v-for="(topic, index) in filteredTopics" :key="topic.id" :topic="topic" />
      </div>
    </div>
  </div>
</template>

<script>
import Header from './Header.vue'
import TopicDisplay from '@/components/TopicDisplay.vue';

export default {
  components: {
    Header,
    TopicDisplay,
  },
  data() {
    return {
      searchTerm: '',
      topics: [
        {
          id: 1,
          title: 'AAPL (Apple Inc.)',
          description: '<b>Information:</b><br> Apple Inc. ist ein multinationales Technologieunternehmen, das für seine Unterhaltungselektronik, Software und Online-Dienste bekannt ist. <br><b>Tags:</b><br> Technologie, Unterhaltungselektronik, Software, iPhone, MacBook, iPad, iOS, macOS',
        },
        {
          id: 2,
          title: 'AAL (American Airlines Group Inc.)',
          description: "<b>Information:</b><br> American Airlines ist eine große amerikanische Fluggesellschaft, die sowohl nationale als auch internationale Flüge anbietet.<br> Fluggesellschaft, Luftfahrt, Reisen, Transport",
        },
        {
          id: 3,
          title: 'AMD (Advanced Micro Devices, Inc.)',
          description: '<b>Information:</b><br> AMD ist ein Halbleiterunternehmen, das CPUs, GPUs und andere Computerhardware herstellt. <br> <b>Tags:</b><br>  Halbleiter, CPU, GPU, Computer-Hardware, Technologie',
        },
        {
          id: 4,
          title: 'C (Citigroup Inc.)',
          description: '<b>Information:</b><br> Citigroup ist eine multinationale Investmentbank und ein Finanzdienstleistungsunternehmen. <br> <b>Tags:</b><br>  Bankwesen, Finanzen, Investitionen, Finanzdienstleistungen',
        },
        {
          id: 5,
          title: 'MRNA (Moderna Inc.)',
          description: '<b>Information:</b><br> Moderna ist ein Biotechnologieunternehmen, das sich auf die Entwicklung von mRNA-basierten Impfstoffen und Therapeutika spezialisiert hat. <br> <b>Tags:</b><br>  Biotechnologie, Pharmazeutika, Impfstoffe, mRNA',
        },
        {
          id: 6,
          title: 'NIO (NIO Inc.)',
          description: '<b>Information:</b><br> NIO ist ein chinesischer Hersteller von Elektrofahrzeugen, der für die Produktion von Elektroautos und verwandter Technologie bekannt ist. <br> <b>Tags:</b><br>  Elektrofahrzeuge, Automobilindustrie, Technologie, EVs, China',
        },
        {
          id: 7,
          title: 'NVDA (NVIDIA Corporation)',
          description: '<b>Information:</b><br> NVIDIA ist ein führendes Technologieunternehmen, das für seine Grafikprozessoren (GPUs) und KI-bezogenen Produkte bekannt ist.<br> <b>Tags:</b><br>  Technologie, GPU, Grafik, Künstliche Intelligenz, Spiele',
        },
        {
          id: 8,
          title: 'SNAP (Snap Inc.)',
          description: '<b>Information:</b><br> Snap Inc. ist ein Technologieunternehmen, das die beliebte Multimedia-Messaging-App Snapchat entwickelt hat.<br> <b>Tags:</b><br>  Soziale Medien, Messaging, Multimedia, Technologie',
        },
        {
          id: 9,
          title: 'SQ (Square Inc.)',
          description: '<b>Information:</b><br> Square ist ein Finanzdienstleistungs- und mobiles Zahlungsunternehmen, das für seine Point-of-Sale-Hardware und seine Cash App bekannt ist. <br> <b>Tags:</b><br>  Finanzdienstleistungen, Mobiler Zahlungsverkehr, Fintech, Technologie',
        },
        {
          id: 10,
          title: 'TSLA (Tesla, Inc.)',
          description: '<b>Information:</b><br>  Tesla ist ein Unternehmen für Elektrofahrzeuge und saubere Energie, das Elektroautos, Solarprodukte und Energiespeicherlösungen herstellt. <br> <b>Tags:</b><br>  Elektrofahrzeuge, Saubere Energie, Automobil, Technologie',
        },
      ],
    };
  },
  computed: {
    filteredTopics() {
      // Filter topics based on the search term
      const searchTerm = this.searchTerm.toLowerCase();
      return this.topics.filter(topic =>
        topic.title.toLowerCase().includes(searchTerm) ||
        topic.description.toLowerCase().includes(searchTerm)
      );
    },
  },
};
</script>

<style>
.background {
  height: 100%;
  width: 100%;
  background-color: #F5F5F5;
}

.subhead {
  margin-top: 130px;
  width: 100%;
  justify-content: center;
  display: flex;
}

.uberschrift {
  flex: 0 0 auto;
  width: 994px;
  height: 100px;
  display: flex;
  align-items: flex-end;
  flex-direction: column;
  justify-content: center;
}

.home-logo {
  font-size: 2em;
  align-self: flex-start;
  font-family: "Lato";
  font-weight: bold;
  margin-left: var(--dl-space-space-unit);
  text-transform: uppercase;
}

.topics-container {
  display: flex;
  max-width: 100%;
  padding: 20px;
  margin-left: 0px;
  overflow-y: auto; /* Add this line to enable vertical scrolling */
  max-height: 60%;
}
</style>