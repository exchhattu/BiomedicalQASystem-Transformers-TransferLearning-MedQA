
from pytorch_transformers import XLNetModel, XLNetTokenizer
from pytorch_transformers import AdamW

import collections

"""
Few functions are borrowed from pytorch_transform. They are primarily for 
tokenization.  https://github.com/rusiaaman/pytorch-transformers
"""

class InputData:

    def __init__(self):
        self._tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)

    def clean_input(self, qa_pair):
        """
        This function is inspired from pytorch_transform example

        """
        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False
        
        paragraph_text= qa_pair._context
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        orig_answer_text = qa_pair._answer # answer["text"]
        answer_offset = qa_pair._answer_start # answer["answer_start"]
        answer_length = len(orig_answer_text)
        start_position = char_to_word_offset[answer_offset]
        end_position = char_to_word_offset[answer_offset + answer_length - 1]
        qa_pair.update(answer_length, start_position, end_position, doc_tokens)


    def _check_is_max_context(self, doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""

        # Because of the sliding window approach taken to scoring documents, a single
        # token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C. We only
        # want to consider the score with "maximum context", which we define as
        # the *minimum* of its left and right context (the *sum* of left and
        # right context will always be the same, of course).
        #
        # In the example the maximum context for 'bought' would be span C since
        # it has 1 left context and 3 right context, while span B has 4 left context
        # and 0 right context.
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index
        return cur_span_index == best_span_index

    def _improve_answer_span(self, doc_tokens, input_start, input_end, tokenizer,
                             orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""

        # The SQuAD annotations are character based. We first project them to
        # whitespace-tokenized words. But then after WordPiece tokenization, we can
        # often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).". However
        # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
        # the exact answer, 1895.
        #
        # However, this is not always possible. Consider the following:
        #
        #   Question: What country is the top exporter of electornics?
        #   Context: The Japanese electronics industry is the lagest in the world.
        #   Answer: Japan
        #
        # In this case, the annotator chose "Japan" as a character sub-span of
        # the word "Japanese". Since our WordPiece tokenizer does not split
        # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
        # in SQuAD, but does happen.
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))
       
        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)

    def tokenize_input(self, qa_pair):
        query_tokens = self._tokenizer.tokenize(qa_pair._question)
        # print(qa_pair._question, query_tokens)

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens    = []
        for (i, token) in enumerate(qa_pair._doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = self._tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        is_training = True
        max_seq_length = 1200 
        cls_token_at_end=False
        cls_token_segment_id=0
        cls_token='[CLS]' 
        sep_token='[SEP]'
        pad_token = 0
        sequence_a_segment_id=0
        sequence_b_segment_id=1
        cls_token_segment_id=0
        pad_token_segment_id=0
        mask_padding_with_zero=True
        doc_stride = 128

        if is_training and not qa_pair._is_impossible:
            tok_start_position = orig_to_tok_index[qa_pair._start_position]
            if qa_pair._end_position < len(qa_pair._doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[qa_pair._end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position, tok_end_position) = \
                        self._improve_answer_span(all_doc_tokens, tok_start_position, 
                                                    tok_end_position, self._tokenizer, qa_pair._answer)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        #
        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = []
            
            # CLS token at the beginning
            if not cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = 0
            
            # Query
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)
           
            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)

            # Paragraph
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                
                is_max_context = self._check_is_max_context(doc_spans, doc_span_index,
                                                            split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(sequence_b_segment_id)
                p_mask.append(0)

            paragraph_len = doc_span.length

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)
            
            # CLS token at the end
            if cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = len(tokens) - 1  # Index of classification token
            
            input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            
            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)
            
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            span_is_impossible = qa_pair._is_impossible
            start_position = None
            end_position = None
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                #                 # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
                    
            if is_training and span_is_impossible:
                start_position = cls_index
                end_position = cls_index
        # return input_ids
        return (input_ids, input_mask, segment_ids, cls_index, p_mask, start_position, end_position)

        # unique_id=unique_id,
        # example_index=example_index,
        # doc_span_index=doc_span_index,
        # tokens=tokens,
        # token_to_orig_map=token_to_orig_map,
        # token_is_max_context=token_is_max_context,
        # input_ids=input_ids,
        # input_mask=input_mask,
        # segment_ids=segment_ids,
        # cls_index=cls_index,
        # p_mask=p_mask,
        # paragraph_len=paragraph_len,
        # start_position=start_position,
        # end_position=end_position,
        # is_impossible=span_is_impossible))

