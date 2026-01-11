## 2.1 `unicode1`

1. What Unicode character does chr(0) return?  
--> '\x00'

2. How does this character’s string representation (__repr__()) differ from its printed representation?  
--> When printed, it is an empty character.

3. What happens when this character occurs in text?
--> When printed, it behaves as an empty character. But when you look at the string, you can see the added bytes.  

## 2.2 `unicode2`

1. What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32?  
--> Because in UTF-16 and UTF-32, you always need at least 2 and 4 bytes to represent one character, respectively. However, this is a waste of space and sequence, as most characters can be represented with 1 byte, as UTF-8 does and only need multiple bytes when representing larger numbers.

2. Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.  

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```

--> This code assumes every character is represented by 1 byte, and this assumption would fail fairly quickly when it comes across any non-ascii characters. Example: If you provide "uğur" as an input, you get a `UnicodeDecodeError` exception:
`UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc4 in position 0: unexpected end of data`

3. Give a two byte sequence that does not decode to any Unicode character(s).  
--> According to the specification, there are 66 non-character byte sequences, such as `1111 1101 1101 0000`.

## 2.4 BPE Tokenizer Training

```python
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
import regex as re
re.findall(PAT, "some text that i'll pre-tokenize")
```
```bash
Out[111]: ['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```