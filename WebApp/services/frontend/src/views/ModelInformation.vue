<template>
  <div>
    <Header />
    <div class="background">
      <div class="subhead">
        <div class="uberschrift">
          <span class="home-logo">Informationen zu den Modellen</span>
        </div>
      </div>
      <div class="topics-container">
        <TopicDisplay
          v-for="(topic, index) in topics"
          :key="topic.id"
          :topic="topic"
          :image-url="getTopicImageUrl(topic)"
          :image-max-width="'800px'"
          :image-max-height="'auto'"
        />
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
      topics: [
        {
          id: 1,
          title: 'Transformer',
          description: '<b>Input:</b><br> Einen Startzeitpunkt und einen Endzeitpunkt (bis wohin predicted werden soll). <br><b>Output:</b><br> vorhergesagten Kursänderungen der Aktien bis zum angegebenen Endzeitpunkt.',
          image: '/Transformer.png'
        },
        {
          id: 2,
          title: 'CNN',
          description: "<b>Input:</b><br> 30x6 Daten Matrix, wobei 30 Tageswerte und 7 die Features sind (GoldPreis und verschiedene ETFs) (in der Datengrundlage ggb.).<br> Dieser Input wird transformiert in 7x30x30 array => somit ist der fertig Model Input = 6x30x30; <br><b>Output:</b><br>Ein Skalar, welcher die Kurs Veränderung nach 5 Tagen predicted."
        },
        {
          id: 3,
          title: 'ANN',
          description: '<b>Input:</b><br> Forecast Horizon. <br> <b>Output:</b><br>  Dictionary, mit Angabe, welche Modelle welche Prognosen abgeben.',
          image: '' // No image for this topic
        },
        {
          id: 4,
          title: 'LSTM',
          description: "Das LSTM-Modell ist eine spezielle Art von rekurrentem neuronalem Netzwerk, das sich besonders gut für die Vorhersage von Zeitreihendaten eignet, wie sie im Finanzmarkt vorzufinden sind. LSTMs zeichnen sich durch ihre Fähigkeit aus, sowohl kurzfristige als auch langfristige Abhängigkeiten in den Daten zu erkennen und zu nutzen. Diese Eigenschaft ist besonders vorteilhaft für die Vorhersage von Aktienkursbewegungen, da sie eine umfassende Berücksichtigung historischer Daten ermöglicht, ohne dass ein explizites Feature-Engineering erforderlich ist. Die Stärke der LSTM-Modelle liegt in ihrer Struktur, die aus Zellen, Gates und Zuständen besteht, die es dem Modell ermöglichen, Informationen über lange Zeiträume zu speichern und zu vergessen."
        },

      ],
    };
  },
  methods: {
    getTopicImageUrl(topic) {
      return topic.image || ''; // Return the image URL if it exists, otherwise return an empty string
    }
  }
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
  flex-wrap: wrap;
  max-width: 100%;
  padding: 20px;
  margin-left: 0px;
  max-height: 70%;
}

</style>