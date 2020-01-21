
"""
https://github.com/huggingface/transformers/blob/master/examples/run_squad.py
"""
import torch
from pytorch_transformers import XLNetConfig, XLNetModel, XLNetTokenizer, XLNetForQuestionAnswering
# from pytorch_transformers import XLNetModel, XLNetTokenizer
from pytorch_transformers.optimization import WarmupConstantSchedule
from pytorch_transformers import AdamW

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)


from transformers.data.processors.squad import SquadResult
from transformers.data.metrics import squad_metrics
from transformers import get_linear_schedule_with_warmup

from tqdm import tqdm, trange

class QaDataModel:

    """
    For training and testing a given data
    """

    def __init__(self, train_dataset):
        self.init_params('xlnet', f_lr=2e-5, f_eps = 1e-8)
        self.make_qa_model(train_dataset, epochs = 4 )

    def init_params(self, model_name, f_lr=2e-5, f_eps = 1e-8):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        MODEL_CLASSES = { "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer) }
        # self._config, self._model_class, self._tokenizer = MODEL_CLASSES[model_name]
        self._tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
        self._config = XLNetConfig.from_pretrained('xlnet-base-cased', do_lower_case=True)
        self._model = XLNetForQuestionAnswering.from_pretrained('xlnet-base-cased', config=self._config)

        param_optimizer = self._model.named_parameters()

        # no_decay = ['bias', 'gamma', 'beta']
        no_decay = ['bias', 'LayerNorm.weight']
        weight_decay = 0.0 # Author's default parameter
        optimizer_grouped_parameters = [
                  {'params': [p for n, p in self._model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                  {'params': [p for n, p in self._model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                  ]
        warmup_steps = 0.0
        self._optimizer = AdamW(optimizer_grouped_parameters, lr=f_lr, eps=f_eps)


    def make_qa_model(self, train_dataset, epochs = 4):
        train_batch_size = 8 # args.per_gpu_train_batch_size * max(1, args.n_gpu) # taken from ...
        train_sampler = RandomSampler(train_dataset) # if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        max_step    = 1 # default value in run_squad.py
        t_total     = max_step 
        gradient_accumulation_steps = 1 # default value in run_squad.py
        self._scheduler = get_linear_schedule_with_warmup(
                            self._optimizer, num_warmup_steps=0 , # default value in run_squad.py 
                            num_training_steps=t_total) # depending upon max_step; however default was chosen 

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0

        self._model.zero_grad()
        # set random seed.
        
        for _ in trange(epochs, desc="Iteration"):
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self._model.train()
                # what is args.device
                batch = tuple(t.to(self._device) for t in batch)

                inputs = {'input_ids':       batch[0],
                          'attention_mask':  batch[1], 
                          'token_type_ids':  batch[2],  
                          'start_positions': batch[3], 
                          'end_positions':   batch[4],
                          'cls_index':       batch[5],
                          'p_mask':          batch[6] }
                outputs = self._model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
                loss.backward()
                tr_loss += loss.item()
                
                # this function does not have effect unless gradient_accumulation_steps > 1
                # however, it was set to 1 in original code.
                # if (step + 1) % 1 == 0:

                self._optimizer.step()
                self._model.zero_grad()
                global_step += 1

                # validation
                # get validation data 
                # 8 * 1
                eval_batch_size = 8 # default suggested by author  # args.per_gpu_eval_batch_size * max(1, args.n_gpu)
                # Note that DistributedSampler samples randomly
                eval_sampler    = SequentialSampler(train_dataset)
                eval_dataloader = DataLoader(train_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    self._model.eval()
                    batch = tuple(t.to(args.device) for t in batch)

                    with torch.no_grad():
                        inputs = { "input_ids": batch[0],
                                   "attention_mask": batch[1],
                                   "token_type_ids": batch[2],
                                   "cls_index": batch[4], 
                                   "p_mask": batch[5]
                                  }
                        example_indices = batch[3]
                        outputs = self._model(**inputs)

                for i, example_index in enumerate(example_indices):
                #         eval_feature = features[example_index.item()]
                #         unique_id = int(eval_feature.unique_id)

                #         output = [to_list(output[i]) for output in outputs]
                #         if len(output) >= 5:
                #             start_logits = output[0]
                #             start_top_index = output[1]
                #             end_logits = output[2]
                #             end_top_index = output[3]
                #             cls_logits = output[4]
                #           
                #             result = SquadResult(
                #             unique_id,
                #             start_logits,
                #             end_logits,
                #             start_top_index=start_top_index,
                #             end_top_index=end_top_index,
                #             cls_logits=cls_logits,
                #             )








    def prepare_qa_model(self, epochs = 4):
        # Store our loss and accuracy for plotting
        self._train_loss_set = []
        # Number of training epochs (authors recommend between 2 and 4)
        # epochs = 4
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
   
