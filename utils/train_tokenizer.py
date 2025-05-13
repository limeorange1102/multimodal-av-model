import os
from glob import glob
import sentencepiece as spm

def train_tokenizer_from_txt_folder(txt_folder, model_prefix='utils/tokenizer800', vocab_size=1300):
    # 1. 모든 .txt 파일 경로 수집
    txt_files = glob(os.path.join(txt_folder, "*.txt"))
    if len(txt_files) == 0:
        raise ValueError(f"❌ {txt_folder} 폴더에 .txt 파일이 없습니다!")

    print(f"📂 {len(txt_files)}개의 자막 파일에서 토크나이저 학습을 시작합니다...")

    # 2. 쉼표로 경로 연결
    input_files = ",".join(txt_files)

    # 3. SentencePiece tokenizer 학습
    spm.SentencePieceTrainer.train(
        input=input_files,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='unigram',
        character_coverage=1.0,
        user_defined_symbols=['<blank>']
    )

    print(f"✅ 학습 완료: {model_prefix}.model / {model_prefix}.vocab")

# train_tokenizer_from_txt_folder("processed_dataset/text")
