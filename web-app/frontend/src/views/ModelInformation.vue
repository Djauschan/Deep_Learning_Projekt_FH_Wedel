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
          description: '<b>CNN Restrictions:</b><br> Daily Predictions, only implemented for: AAL, AAPL, AMD, C',
          image: '/CNN.png'
        },
        {
          id: 3,
          title: 'ANN',
          description: 'Random Forest ist ein Ensemble-Modell, das aus vielen unabhängigen Entscheidungsbäumen besteht und durchschnittliche Vorhersagen liefert. Anstatt sich auf ein einzelnes Modell zu stützen, nutzt Random Forest die Stärken vieler Bäume, um präzisere und robustere Ergebnisse zu erzielen.<br>Das Gradient Boosting Machines ist ebenfalls eine Ensemble-Modelle, die jedoch ihre Fehler sequenziell durch aufeinanderfolgende Bäume korrigieren. Jeder Baum im Gradient Boosting Machines wird nacheinander gebaut, wobei er versucht, die Fehler seines Vorgängers zu korrigieren. Dies unterscheidet sich somit vom Random Forest, welche unabhängigen Bäume erstellt.<br>Die Eingabevariablen setzen sich aus Merkmalen zusammen, die mittels Feature-Engineering aus dem eigenen Datensatz abgeleitet wurden. Dabei wurden unterschiedliche Features entwickelt. Dazu zählt das DateTime-Feature, welches zeitliche Informationen wie Wochentage, Stunden und Monate erfasst. Des weiteren ein Merkmal, das prozentuale Veränderungen abbildet, sowie Lag-Features, welche Verzögerungsindikatoren darstellen und zeitlich verschobene Datenpunkte erfassen.<br>Der Output der Modelle liefert Prognosen für zukünftige Aktienschlusskurse, wobei das jeweilige Vorhersageintervall von der Auswahl des Modells abhängt. Das Daily-Modell prognostiziert die Schlusskurse der kommenden Tage. Beim 2-Stunden-Modell werden Vorhersagen in Zwei-Stunden-Intervallen getroffen, während das minütliche Modell alle 20 Minuten Vorhersagen bereitstellt.',
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

</style>../../../../../web-app/frontend/src/views/Header.vue
