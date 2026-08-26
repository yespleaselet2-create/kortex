import os
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing


class KortexTokenizer:
    PAD = "[PAD]"
    BOS = "[BOS]"
    EOS = "[EOS]"
    UNK = "[UNK]"

    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size
        self.tokenizer = None

    def train_from_files(self, files, save_dir):
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=[self.PAD, self.BOS, self.EOS, self.UNK],
            show_progress=True,
        )
        self.tokenizer = Tokenizer(BPE(unk_token=self.UNK))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tokenizer.train(files, trainer)
        self.tokenizer.post_processor = TemplateProcessing(
            single=f"{self.BOS}:0 $A:0 {self.EOS}:0",
            special_tokens=[(self.BOS, 1), (self.EOS, 2)],
        )
        os.makedirs(save_dir, exist_ok=True)
        self.tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
        print(f"Tokenizer saved to {save_dir} with vocab size {self.tokenizer.get_vocab_size()}")

    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)
        print(f"Tokenizer loaded from {path} with vocab size {self.tokenizer.get_vocab_size()}")

    def encode(self, text, max_length=1024, padding=False, truncation=True):
        enc = self.tokenizer.encode(text)
        ids = enc.ids[:max_length] if truncation else enc.ids
        if padding and len(ids) < max_length:
            ids = ids + [self.tokenizer.token_to_id(self.PAD)] * (max_length - len(ids))
        return ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    @property
    def pad_id(self):
        return self.tokenizer.token_to_id(self.PAD)

    @property
    def bos_id(self):
        return self.tokenizer.token_to_id(self.BOS)

    @property
    def eos_id(self):
        return self.tokenizer.token_to_id(self.EOS)

    @property
    def vocab_size(self):
        if self.tokenizer:
            return self.tokenizer.get_vocab_size()
        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, v):
        self._vocab_size = v
