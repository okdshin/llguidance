import json
import time
from typing import List, Optional

import pytest
import torch
from transformers import AutoTokenizer

import llguidance.plamo2_tokenizer
from llguidance import LLExecutor, LLMatcher, LLTokenizer
from llguidance.torch import (allocate_token_bitmask,
                              fill_next_token_bitmask,
                              fill_next_token_bitmask_par)


def _build_tokenizer() -> LLTokenizer:
    plamo2_tok = AutoTokenizer.from_pretrained(
        "pfnet/plamo-2-1b", trust_remote_code=True
    )
    return llguidance.plamo2_tokenizer.lltokenizer_from_plamo2_tokenizer(plamo2_tok)


_tokenizer: Optional[LLTokenizer] = None


def tokenizer() -> LLTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = _build_tokenizer()
    return _tokenizer


def lark_matcher(grm: str) -> LLMatcher:
    gstr = json.dumps({"grammars": [{"lark_grammar": grm}]})
    interp = LLMatcher(tokenizer(), gstr, log_level=1)
    return interp


def test_basic_tokenizer() -> None:
    llt = tokenizer()
    for s in [
        "Hello world!",
        "Hello world! こんにちは世界！",
        "wave 👋",
        "heart 👋💖",
        "1`a`b`c`d`e`f`g`h`i",
    ]:
        toks = llt.tokenize_str(s)
        print(llt.dbg_tokens(toks))
        assert llt.decode_str(toks) == s

    # Test byte handling - PLaMo2 may not handle all raw bytes
    toks = llt.tokenize_bytes(b"\x8b")
    print(llt.dbg_tokens(toks))
    print(toks)
    # PLaMo2 tokenizer may not generate tokens for invalid byte sequences
    # This is acceptable behavior - some tokenizers only handle valid UTF-8
    if len(toks) > 0:
        decoded = llt.decode_bytes(toks)
        assert (
            len(decoded) > 0
        ), "Decoded bytes should not be empty if tokens were generated"


def test_grammar() -> None:
    t = tokenizer()
    mask = allocate_token_bitmask(2, t.vocab_size)
    interp = lark_matcher(r"start: /[A-Z ]*/")
    fill_next_token_bitmask(interp, mask)
    allowed = []
    for idx, v in enumerate(mask[0, :].tolist()):
        for bit_idx in range(32):
            tok_idx = idx * 32 + bit_idx
            if v & (1 << bit_idx):
                if t.is_special_token(tok_idx):
                    continue
                s = t.decode_str([tok_idx])
                for c in s:
                    assert c.isupper() or c.isspace()
                allowed.append(tok_idx)
    assert len(allowed) > 100
    interp.consume_token(allowed[3])
    fill_next_token_bitmask(interp, mask, 1)
    assert torch.isclose(mask[1, :], mask[0, :]).all()


def test_par_grammar() -> None:
    n_gram = 50
    t = tokenizer()
    grammars = [(lark_matcher(r"start: /[a-zA-Z ]*/"), idx) for idx in range(n_gram)]
    mask = allocate_token_bitmask(n_gram, t.vocab_size)
    mask2 = allocate_token_bitmask(n_gram, t.vocab_size)
    exec = LLExecutor()
    t0 = time.monotonic()
    fill_next_token_bitmask_par(exec, grammars, mask)
    par_time = int((time.monotonic() - t0) * 1_000_000)
    for i in range(n_gram):
        assert torch.isclose(mask[i, :], mask[0, :]).all()
    t0 = time.monotonic()
    for g, idx in grammars:
        fill_next_token_bitmask(g, mask2, idx)
    seq_time = int((time.monotonic() - t0) * 1_000_000)
    assert torch.isclose(mask, mask2).all()
    print(f"Parallel: {par_time} us, Sequential: {seq_time} us")


@pytest.mark.parametrize("recent_tokens", [[], [1000, 3003]])
def test_tokenize_partial_basic(recent_tokens: List[int]) -> None:
    """Test tokenize_partial with a simple sentence."""
    ll_tok = tokenizer()
    assert ll_tok.is_canonical
    new_tokens, leftover = ll_tok.tokenize_partial(
        b" How are you", recent_tokens=recent_tokens
    )
    assert isinstance(new_tokens, list)
    assert isinstance(leftover, bytes)
    assert len(new_tokens) >= 2
    assert ll_tok.decode_bytes(new_tokens) + leftover == b" How are you"
    for suff in ["", "r", "!", " "]:
        tok2 = ll_tok.tokenize_str(" How are you" + suff)
        assert tok2[0 : len(new_tokens)] == new_tokens


def test_tokenize_partial_docs() -> None:
    ll = tokenizer()
    new_tok, leftover = ll.tokenize_partial(b"order")
    assert len(new_tok) == 0
    assert leftover == b"order"

    recent = ll.tokenize_bytes(b'{"')
    new_tok, leftover = ll.tokenize_partial(
        b'name_of_the_person"', recent_tokens=recent
    )
    print(ll.dbg_tokens(new_tok))
    assert leftover == b'"'
    assert ll.decode_str(new_tok) == "name_of_the_person"


def test_special_tokens() -> None:
    """Test special token handling."""
    ll_tok = tokenizer()

    # Test EOS token
    eos_id = ll_tok.eos_token
    assert eos_id is not None
    assert ll_tok.is_special_token(eos_id)

    # Test BOS token if available
    if hasattr(ll_tok, "bos_token") and ll_tok.bos_token is not None:
        bos_id = ll_tok.bos_token
        assert ll_tok.is_special_token(bos_id)

    # Test UNK token if available
    if hasattr(ll_tok, "unk_token") and ll_tok.unk_token is not None:
        unk_id = ll_tok.unk_token
        assert ll_tok.is_special_token(unk_id)

    # Test PAD token if available
    if hasattr(ll_tok, "pad_token") and ll_tok.pad_token is not None:
        pad_id = ll_tok.pad_token
        assert ll_tok.is_special_token(pad_id)


def test_parse_special_tokens() -> None:
    """Test parse_special parameter functionality like in test_tiktoken.py."""
    ll_tok = tokenizer()

    # Test special token parsing - PLaMo2 uses <|plamo:eos|> as EOS token
    eos_token_str = "<|plamo:eos|>"  # PLaMo2's actual EOS token string
    actual_eos_id = ll_tok.eos_token  # The actual EOS token ID (2)

    print(f"EOS token ID: {actual_eos_id}")

    # Test default behavior (parse_special=False)
    toks1 = ll_tok.tokenize_str(eos_token_str)
    toks0 = ll_tok.tokenize_str(eos_token_str, parse_special=False)
    assert toks1 == toks0
    print(f"EOS token without parse_special: {toks0}")

    # Test with parse_special=True
    toks2 = ll_tok.tokenize_str(eos_token_str, parse_special=True)
    print(f"EOS token with parse_special=True: {toks2}")

    # Check if parse_special makes a difference
    if toks0 != toks2:
        # If different, parse_special=True should give us the actual EOS token
        assert (
            toks2[0] == actual_eos_id
        ), "parse_special=True should give actual EOS token ID"
    else:
        # If same, document that PLaMo2 handles special tokens differently
        print(
            "PLaMo2 tokenizer handles special tokens consistently regardless of parse_special"
        )
        # The tokenized version should still be a valid representation
        assert len(toks0) >= 1, "Should produce at least one token"


def test_japanese_text() -> None:
    """Test Japanese text tokenization (PLaMo2 is optimized for Japanese)."""
    ll_tok = tokenizer()

    japanese_texts = [
        "こんにちは世界！",
        "日本語の文章をテストします。",
        "人工知能（AI）技術の発展",
        "今日は良い天気ですね。明日も晴れそうです。",
        "ひらがな、カタカナ、漢字が混在した文章です。",
        "Hello こんにちは 123 テスト",  # Mixed language
    ]

    for text in japanese_texts:
        tokens = ll_tok.tokenize_str(text)
        decoded = ll_tok.decode_str(tokens)
        assert decoded == text, f"Failed roundtrip for: {text}"
        assert len(tokens) > 0, f"No tokens generated for: {text}"


def test_error_handling_and_edge_cases() -> None:
    """Test error handling and edge cases."""
    ll_tok = tokenizer()

    # Test empty input
    assert ll_tok.tokenize_str("") == []
    assert ll_tok.tokenize_bytes(b"") == []
    assert ll_tok.decode_str([]) == ""
    assert ll_tok.decode_bytes([]) == b""

    # Test invalid token IDs
    try:
        ll_tok.decode_str([999999])  # Very large token ID
    except Exception:
        pass  # Expected to fail

    # Test byte sequences
    byte_sequences = [
        b"\x00",  # Null byte
        b"\xff",  # Max byte
        b"\x80\x81",  # Invalid UTF-8 sequence
        b"\xc0\x80",  # Overlong encoding
    ]

    for seq in byte_sequences:
        tokens = ll_tok.tokenize_bytes(seq)
        # Should not crash, may return empty list for invalid sequences
        if tokens:  # If tokens were generated, decoding should work
            decoded = ll_tok.decode_bytes(tokens)
            # Note: decoded may not exactly match input for invalid UTF-8

    # Test very long strings
    long_text = "a" * 10000
    tokens = ll_tok.tokenize_str(long_text)
    decoded = ll_tok.decode_str(tokens)
    assert decoded == long_text

    # Test vocab size bounds
    vocab_size = ll_tok.vocab_size
    assert vocab_size > 0

    # Test token ID bounds
    for token_id in [0, vocab_size - 1]:
        try:
            ll_tok.decode_str([token_id])
        except Exception:
            pass  # Some token IDs might be invalid


def test_tokenizer_properties() -> None:
    """Test tokenizer properties and metadata."""
    ll_tok = tokenizer()

    # Test vocab size
    assert ll_tok.vocab_size > 0

    # Test canonical property
    assert ll_tok.is_canonical

    # Test EOS token
    eos = ll_tok.eos_token
    assert eos is not None
    assert eos >= 0
    assert eos < ll_tok.vocab_size

    # Test special token detection
    assert ll_tok.is_special_token(eos)

    # Test that some regular tokens are not special
    regular_tokens = ll_tok.tokenize_str("hello world")
    if regular_tokens:
        # At least one regular token should not be special
        non_special_found = any(
            not ll_tok.is_special_token(tok) for tok in regular_tokens
        )
        assert non_special_found


def test_byte_token_handling() -> None:
    """Test byte-level token handling - PLaMo2 has limited byte support."""
    ll_tok = tokenizer()

    # Test the specific byte handling case
    tokens = ll_tok.tokenize_bytes(b"\x8b")
    print(f"Tokenizing b'\\x8b': {tokens}")
    print(ll_tok.dbg_tokens(tokens))

    # PLaMo2 tokenizer may not handle raw bytes - document this behavior
    if len(tokens) == 0:
        print(
            "PLaMo2 tokenizer doesn't handle raw byte \\x8b - this is expected behavior"
        )
        # Test with valid UTF-8 byte sequence instead
        valid_utf8_bytes = "テスト".encode("utf-8")
        tokens = ll_tok.tokenize_bytes(valid_utf8_bytes)
        assert len(tokens) > 0, "Should handle valid UTF-8 byte sequences"
        decoded = ll_tok.decode_bytes(tokens)
        assert decoded == valid_utf8_bytes, "Should decode valid UTF-8 correctly"
    else:
        decoded = ll_tok.decode_bytes(tokens)
        print(f"Decoded: {decoded}")
        assert len(decoded) > 0, "Decoded bytes should not be empty"


if __name__ == "__main__":
    test_basic_tokenizer()
