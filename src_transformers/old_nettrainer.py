import numpy as np
import torch
import torch.utils.data as torchData
from loggers.logger import Logger


class NetTrainer():
    """
    Diese Klasse fuehrt das Training durch.
    """

    def __init__(self, model, dataset, criterion,
                 seed=69, gpu=True,
                 name="", val_split=0.2, batch_size=1, dataholder_str=""):

        # Parameter initialisierung
        self.model = model
        self.dataset = dataset
        self.criterion = criterion
        self.seed = seed
        self.gpu = gpu and torch.cuda.is_available()

        # RNG Seed setzen
        torch.manual_seed(self.seed)
        if self.gpu:
            torch.cuda.manual_seed(self.seed)

        # Erzeugen des Loggers
        self.logger = Logger(name)

        # Erste Informationen loggen
        self.logger.summary("dataholder", dataholder_str)
        self.logger.model_text(self.model)
        self.logger.summary("seed", self.seed)
        # Datenset vorbereiten
        self.train_loader, self.validation_loader = self.prepare_dataset(
            val_split, batch_size)
        # Initialisierung der Grafikkarte
        self.gpu_setup()

    def prepare_dataset(self, validation_split, batch_size):
        """
        Dataset splitten und in Validation und Trainingset und in Dataloader stecken.
        """
        dataset_size = len(self.dataset)
        validation_size = int(np.floor(validation_split * dataset_size))
        train_size = dataset_size - validation_size

        train_dataset, validation_dataset = torchData.random_split(self.dataset,
                                                                   [train_size,
                                                                    validation_size])

        train_loader = torchData.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True)
        validation_loader = torchData.DataLoader(validation_dataset,
                                                 batch_size=batch_size)

        return train_loader, validation_loader

    def gpu_setup(self):
        """
        FUnktion die das Model und das Criterion auf die GPU laedt.
        """
        if self.gpu:
            print("[TRAINER]: Write model and criterion on GPU")
            self.model.to('cuda')
            self.criterion.to('cuda')

        # Log if cpu or gpu is used
        self.logger.log_string("device usage", ("GPU" if self.gpu else "CPU"))

    def calc_step(self, minibatch, target):
        """
        Funktion zum berechnen eines Trainingschritts.
        """
        # Reset Optimizer
        self.optimizer.zero_grad()

        # Wenn noetig Daten auf die GPU laden
        if self.gpu:
            minibatch = minibatch.to('cuda')
            target = target.to('cuda')

        # Forward step
        prediction = self.model.forward(minibatch)

        # Loss Berechnung
        loss = self.criterion(prediction, target.long())
        
        # Backpropagation
        loss.backward()
        
        # Optimazation step
        self.optimizer.step()

        return loss.item()

    def calc_epoch(self):
        """
        Methode zum Berechnen eines ganzen Epochs.
        Dabei wird das Dataset ggf. mehrere Male durchlagen, je nach dem
        inflation Parameter. Da eine Augmentation stattfindet, wird das Dataset
        praktisch aufgebleht.
        """
        epoch_loss = 0.0
        step_count = 0

        self.model.train()
        self.dataset.set_validation(False)
        self.dataset.set_rng_back()

        for _ in range(self.inflation):
            # Durchlaufe das ganze Dataset
            for (mini_batch, target) in self.train_loader:
                epoch_loss += self.calc_step(mini_batch, target)
                step_count += 1

        return (epoch_loss / step_count)

    def train(self, epochs, optimizer, patience=0, inflation=1):
        """
        Methode fuer das ganze Training
        """
        # Log den Start des Trainings
        self.logger.train_start()

        self.optimizer = optimizer
        self.inflation = inflation

        # Daten fuer Early stopping
        min_loss = float('inf')
        cur_patience = 0
        finish_reason = 'Training did not start'

        # Durchlaufe folgende Schritte fuer jeden Epoch
        for epoch in range(epochs):
            # Try block um das Training mit Ctrl+c abbrechen zu können
            try:

                train_loss = self.calc_epoch()
                # Log Training Loss
                self.logger.train_loss(train_loss, epoch)

                # Nach Berechnung eines Epochs im Training Validiere
                # Dies kann auch seltener passieren z.B. alle 10 Epochs
                # if(epoch % 10) == 0:
                validation_loss = self.validate(epoch)

                # Early stopping
                if (min_loss > validation_loss):
                    min_loss = validation_loss
                    cur_patience = 0
                    # Speicher das aktuell beste Netz
                    self.logger.save_net(self.model)
                else:
                    if (patience > 0):
                        cur_patience += 1
                        if (cur_patience == patience):
                            finish_reason = 'Training finished because of early stopping'
                            break

            except KeyboardInterrupt:
                # Fuer den Fall wir wollen das Training haendisch abbrechen
                finish_reason = 'Training interrupted by user'
                break
            else:
                finish_reason = 'Training finished normally'

        # Log finish
        self.logger.train_end(finish_reason)

        # Training Cleanup
        self.logger.close()

    def validate(self, epoch=0):
        """
        Berechnung einer Validierung.
        Der Epoch wird uebergeben um ggf. diesen zu loggen.
        """

        # Setze das Model in den Eval Modus
        self.model.eval()
        # Setze das Dataset in den Validierungszustand
        self.dataset.set_validation(True)

        valid_loss = 0.0
        steps = 0

        # Der Gradient soll nicht Berechnet werden
        with torch.no_grad():
            for (minibatch, target) in self.validation_loader:
                # Ggf. auf GPU Laden
                if self.gpu:
                    minibatch = minibatch.to('cuda')
                    target = target.to('cuda')

                prediction = self.model.forward(minibatch)
                loss = self.criterion(prediction, target.long())

                valid_loss += loss.sum().item()

                steps += 1

        # Log validateion_loss
        self.logger.val_loss((valid_loss / steps), epoch)
        # Log das Image wie das Dataset gerade Klassifieziert wird
        self.logger.save_cur_image(self.model, epoch,
                                   self.dataset.data_input,
                                   self.dataset.data_output)

        return (valid_loss / steps)
