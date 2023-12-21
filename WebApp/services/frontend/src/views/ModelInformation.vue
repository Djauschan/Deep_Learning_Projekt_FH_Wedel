<template>
    <Header />
    <div class="background">
        <div class="subhead">
            <div class="uberschrift">
                <span class="home-logo">Informationen zu den Modellen</span>
            </div>
        </div>
        <div id="topic">
            <div class="row" v-for="(topic, index) in topics" :key="topic.id">
                <TopicDisplay :topic="topic" />
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
        },
        {
          id: 2,
          title: 'CNN',
          description: "<b>Input:</b><br> 30x6 Daten Matrix, wobei 30 Tageswerte und 7 die Features sind (GoldPreis und verschiedene ETFs) (in der Datengrundlage ggb.).<br> Dieser Input wird transformiert in 7x30x30 array => somit ist der fertig Model Input = 6x30x30; <br><b>Output:</b><br>Ein Skalar, welcher die Kurs Veränderung nach 5 Tagen predicted.",
        },
        {
          id: 3,
          title: 'CNN',
          description: '<b>Input:</b><br> Forecast Horizon. <br> <b>Output:</b><br>  Dictionary, mit Angabe, welche Modelle welche Prognosen abgeben.',
        },
      ],
      };
    },
  };
  </script>
<style>

#topic { 
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  align-items: center;
  min-height: 40vh;
  padding: 20px;
}

.topics-container {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-around;
  max-width: 400px;
}

#row {
  width: calc(33.33% - 20px);
  margin: 10px;
}

.subhead{
  margin-top: 130px;
  width: 100%;
  justify-content: center;
  display: flex;
}

.background{
  height: 100%;
  width: 100%;
  background-color: #306197;
}

.uberschrift {
  flex: 0 0 auto;
  width: 994px;
  height: 100px;
  display: flex;
  box-shadow: 5px 5px 10px 5px #000000;
  align-items: flex-end;
  flex-direction: column;
  justify-content: center;
  background-color: #ffffff;
}

.home-logo {
  font-size: 2em;
  align-self: flex-start;
  font-family: "Lato";
  font-weight: bold;
  margin-left: var(--dl-space-space-unit);
  text-transform: uppercase;
}
</style>