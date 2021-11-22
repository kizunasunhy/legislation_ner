import json
import copy

class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, src, tar, subject):
        self.guid = guid
        self.src = src
        self.tar = tar
        self.subject = subject
    def __repr__(self):
        return str(self.to_json_string())
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
    
class InputFeature(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, start_ids, end_ids, subjects):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_ids = start_ids
        #self.input_len = input_len
        self.end_ids = end_ids
        self.subjects = subjects

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
    
class Processor():
    """Processor for the chinese ner data set."""
    def __init__(self, label_dict, label_dict_rev):
        self.label_dict = label_dict
        self.label_dict_rev = label_dict_rev
    
    def get_train_examples(self, train_texts, train_tags):
        """See base class."""
        return self._create_examples(train_texts, train_tags, "train")

    def get_val_examples(self, val_texts, val_tags):
        """See base class."""
        return self._create_examples(val_texts, val_tags, "val")

    def _create_examples(self, texts, tags, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, tt) in enumerate(zip(texts, tags)):
            text, tag = tt
            src = text[0]
            tar = text[1]
            guid = "%s-%s" % (set_type, i)
            tag_text = [self.label_dict_rev[ta] for ta in tag]
            subject = self.get_entity_bio(tag_text, id2label=self.label_dict_rev)
            examples.append(InputExample(guid=guid, src=src, tar=tar, subject=subject))
        return examples
    
    def get_entity_bio(self, seq, id2label):
        """Gets entities from sequence.
        note: BIO
        Args:
            seq (list): sequence of labels.
        Returns:
            list: list of (chunk_type, chunk_start, chunk_end).
        Example:
            seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
            get_entity_bio(seq)
            #output
            [['PER', 0,1], ['LOC', 3, 3]]
        """
        chunks = []
        chunk = [-1, -1, -1]
        for indx, tag in enumerate(seq):
            if not isinstance(tag, str):
                tag = id2label[tag]
            if tag.startswith("B-"):
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
                chunk[1] = indx
                chunk[0] = '-'.join(tag.split('-')[1:])
                chunk[2] = indx
                if indx == len(seq) - 1:
                    chunks.append(chunk)
            elif tag.startswith('I-') and chunk[1] != -1:
                _type = '-'.join(tag.split('-')[1:])
                if _type == chunk[0]:
                    chunk[2] = indx

                if indx == len(seq) - 1:
                    chunks.append(chunk)
            else:
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
        return chunks
    
