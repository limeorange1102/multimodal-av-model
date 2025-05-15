import os
from glob import glob
import sentencepiece as spm

def train_tokenizer_from_txt_folder(txt_folder, model_prefix='utils/tokenizer800', vocab_size=1300):
    txt_files = glob(os.path.join(txt_folder, "*.txt"))
    if len(txt_files) == 0:
        raise ValueError(f"❌ {txt_folder} 폴더에 .txt 파일이 없습니다!")

    print(f"📂 {len(txt_files)}개의 자막 파일에서 토크나이저 학습을 시작합니다...")

    input_files = ",".join(txt_files)

    spm.SentencePieceTrainer.train(
        input=input_files,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='char',  # ✅ 글자 단위
        character_coverage=1.0,
        user_defined_symbols=['<blank>', ' ']  # ✅ 공백 포함
    )

    print(f"✅ 학습 완료: {model_prefix}.model / {model_prefix}.vocab")


train_tokenizer_from_txt_folder("processed_dataset/text")
