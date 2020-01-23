



class DataModel:

    """
    For training and testing a given data
    """

    def __init__(self):


    def load_xlnet_data(self):
        self._model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=2)

    def load_qa_data(self. model_name = "xlnet"):
        MODEL_CLASSES = { "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer) }
        self._config, self._model, self._tokenizer = MODEL_CLASSES["xlnet"]

    def init_params(self, f_lr=2e-5):
        param_optimizer = list(self._model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01 },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0 }]
        self._optimizer = AdamW(optimizer_grouped_parameters, lr=f_lr)

    def prepare_model_multilabel_classification(self):
        # Store our loss and accuracy for plotting
        self._train_loss_set = []

        # Number of training epochs (authors recommend between 2 and 4)
        epochs = 4

        # trange is a tqdm wrapper around the normal python range
        for _ in trange(epochs, desc="Epoch"):
            # Training
            # Set our model to training mode (as opposed to evaluation mode)
            self._model.train()

            # Tracking variables
            f_train_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            # Train the data for one epoch
            for step, batch in enumerate(train_dataloader):
                self._optimizer.zero_grad() # Clear out the gradients (by default they accumulate)

                # Add batch to GPU
                # batch = tuple(t.to(device) for t in batch)
                batch = tuple(t for t in batch)
                b_input_ids, b_input_mask, b_labels = batch # Unpack the inputs from our dataloader
                # Forward pass
                outputs = self._model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs[0]
                logits = outputs[1]
                self._train_loss_set.append(loss.item())

                # Backward propagation; Update parameters and take a step using the computed gradient
                loss.backward()
                optimizer.step()

                # Update tracking variables
                self._train_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            print("Train loss: {}".format(f_train_loss/nb_tr_steps))


            # Validation
            # Put model in evaluation mode to evaluate loss on the validation set
            self._model.eval()

            # Tracking variables
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                # Add batch to GPU
                # batch = tuple(t.to(device) for t in batch)
                batch = tuple(t for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch
                # Telling the model not to compute or store gradients, saving memory and speeding up validation
                with torch.no_grad():
                    # Forward pass, calculate logit predictions
                    output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                    logits = output[0]

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1
            print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
   
