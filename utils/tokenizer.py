class Tokenizer:
    def __init__(self, vocab_path: str):
        self.token_to_id = {}
        self.id_to_token = []

        with open(vocab_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token = line.strip().split('\t')[0]  # SentencePiece vocab 형식
                self.token_to_id[token] = idx
                self.id_to_token.append(token)

    def encode(self, text: str):
        # 문자 단위로 토큰 인코딩 (SentencePiece 기반이 아니므로 문자 단위로)
        return [self.token_to_id.get(char, self.token_to_id.get('<unk>', 0)) for char in text]

    def decode(self, ids: list):
        return ''.join([self.id_to_token[i] for i in ids if i < len(self.id_to_token)])

    @property
    def vocab_size(self):
        return len(self.id_to_token)
