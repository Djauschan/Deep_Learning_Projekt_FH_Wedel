import random

class DatasetTransformer(object):
    """
    Classe die verwedenet wird um die Punkte leicht zu verschieben.
    Dies wird Augmentation genannt und kann von wenigen Datums mehrere
    erzeugen.
    """

    def __init__(self, seed, translateH_range, translateH_chance,
                 translateV_range, translateV_chance):
        self.seed = seed
        self.rng = random.Random(self.seed)
        self.translateH_chance = translateH_chance
        self.translateH_range = translateH_range
        self.translateV_chance = translateV_chance
        self.translateV_range = translateV_range

    @classmethod
    def fromDict(cls, data):
        """
        Funktion um einen Konstruktor ueber json zu erzeugen.
        """
        return cls(data["seed"],
                   data["translateH_range"],
                   data["translateH_chance"],
                   data["translateV_range"],
                   data["translateV_chance"])

    def set_rng_back(self):
        """
        Setzt den RNG Seed wieder zurueck um die gleichen Augmentation zu erzeugen.
        """
        self.rng = random.Random(self.seed)

    def __str__(self):
        """
        Gibt eine String Repraesentation der Klasse zurueck.
        """
        return """Transformer:  \n
        Seed: {0}  \n
        TranslationH chance: {1}  \n
        TranslationH range(min, max): {2}  \n
        TranslationV chance: {3}  \n
        TranslationV range(min, max): {4}  \n
        """.format(self.seed, self.translateH_chance, self.translateH_range, self.translateV_chance, self.translateV_range)
        
    def transform(self, input_data):
        """
        Transformiert das input_data.
        """

        if(self.rng.random() < self.translateH_chance):
            transh = self.rng.uniform(self.translateH_range[0], self.translateH_range[1])
        else:
            transh = 0

        if(self.rng.random() < self.translateV_chance):
            transv = self.rng.uniform(self.translateV_range[0], self.translateV_range[1])
        else:
            transv = 0
        
        input_data[0] += input_data[0] * transh
        input_data[1] += input_data[1] * transv

        return input_data
