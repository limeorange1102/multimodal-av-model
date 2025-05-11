import os

class Tokenizer:
    def __init__(self, vocab_path):
        self.token_to_id = {}
        self.id_to_token = []
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token = line.strip().split('\t')[0]  # sentencepiece .vocab format
                self.token_to_id[token] = idx
                self.id_to_token.append(token)

    def encode(self, text):
        # 여기선 한 글자씩 매핑 (subword tokenizer일 경우 바뀌어야 함)
        return [self.token_to_id.get(ch, self.unk_id) for ch in text]

    def decode(self, ids):
        return ''.join([self.id_to_token[i] for i in ids if i < len(self.id_to_token)])

    @property
    def vocab_size(self):
        return len(self.id_to_token)

    @property
    def pad_id(self):
        return self.token_to_id.get('<pad>', 0)

    @property
    def blank_id(self):
        return self.token_to_id.get('<blank>', 0)

    @property
    def unk_id(self):
        return self.token_to_id.get('<unk>', 0)
