# 📘 korean_vocab_utils.py
# 한글 음절 기반 vocab 생성 및 문자열 <-> 인덱스 시퀀스 변환 함수

import unicodedata

# ✅ 완성형 한글 유니코드 범위
BASE_CODE, CHOSUNG, JUNGSUNG = 44032, 588, 28

# 초성, 중성, 종성 리스트
CHOS = [
    "ᄀ", "ᄁ", "ᄂ", "ᄃ", "ᄄ", "ᄅ", "ᄆ",
    "ᄇ", "ᄈ", "ᄉ", "ᄊ", "ᄋ", "ᄌ", "ᄍ",
    "ᄎ", "ᄏ", "ᄐ", "ᄑ", "ᄒ"
]
JUNGS = [
    "ᅡ", "ᅢ", "ᅣ", "ᅤ", "ᅥ", "ᅦ", "ᅧ",
    "ᅨ", "ᅩ", "ᅪ", "ᅫ", "ᅬ", "ᅭ", "ᅮ",
    "ᅯ", "ᅰ", "ᅱ", "ᅲ", "ᅳ", "ᅴ", "ᅵ"
]
JONGS = [
    "", "ᆨ", "ᆩ", "ᆪ", "ᆫ", "ᆬ", "ᆭ",
    "ᆮ", "ᆯ", "ᆰ", "ᆱ", "ᆲ", "ᆳ", "ᆴ",
    "ᆵ", "ᆶ", "ᆷ", "ᆸ", "ᆹ", "ᆺ", "ᆻ",
    "ᆼ", "ᆽ", "ᆾ", "ᆿ", "ᇀ", "ᇁ", "ᇂ"
]

# ⚙️ 음절 문자들을 vocab으로 구성
hangul_syllables = [chr(code) for code in range(0xAC00, 0xD7A4)]

# ⚠️ CTC blank token은 보통 0번 사용
VOCAB = ["<blank>"] + hangul_syllables
char2idx = {ch: i for i, ch in enumerate(VOCAB)}
idx2char = {i: ch for i, ch in enumerate(VOCAB)}

# 🔁 문자열 → 인덱스 시퀀스

def text_to_indices(text):
    return [char2idx.get(ch, 0) for ch in text if ch in char2idx]

# 🔁 인덱스 시퀀스 → 문자열

def indices_to_text(indices):
    return "".join([idx2char[i] for i in indices if i != 0])

# ✅ 테스트용
if __name__ == "__main__":
    test = "바나나"
    idxs = text_to_indices(test)
    print("Encoded:", idxs)
    print("Decoded:", indices_to_text(idxs))
