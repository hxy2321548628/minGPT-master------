"""
bpe is short for Byte Pair Encoder. It translates arbitrary utf-8 strings into
sequences of integers, where each integer represents small chunks of commonly
occuring characters. This implementation is based on openai's gpt2 encoder.py:
https://github.com/openai/gpt-2/blob/master/src/encoder.py
but was mildly modified because the original implementation is a bit confusing.
I also tried to add as many comments as possible, my own understanding of what's
going on.
"""

import os
import json
import regex as re
import requests

import torch

# -----------------------------------------------------------------------------


'''def bytes_to_unicode():
    """
    Every possible byte (really an integer 0..255) gets mapped by OpenAI to a unicode
    character that represents it visually. Some bytes have their appearance preserved
    because they don't cause any trouble. These are defined in list bs. For example:
    chr(33) returns "!", so in the returned dictionary we simply have d[33] -> "!".
    However, chr(0), for example, is '\x00', which looks ugly. So OpenAI maps these
    bytes, into new characters in a range where chr() returns a single nice character.
    So in the final dictionary we have d[0] -> 'Ā' instead, which is just chr(0 + 2**8).
    In particular, the space character is 32, which we can see by ord(' '). Instead,
    this function will shift space (32) by 256 to 288, so d[32] -> 'Ġ'.
    So this is just a simple one-to-one mapping of bytes 0..255 into unicode characters
    that "look nice", either in their original form, or a funny shifted character
    like 'Ā', or 'Ġ', etc.
    """
    # the 188 integers that render fine in their original form and need no shifting
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]  # all integers b in bs will simply map to chr(b) in the output dict
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    n = 0
    for b in range(2**8):
        if b not in bs:
            # if this byte is "ugly" then map it to the next available "nice" character
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    d = dict(zip(bs, cs))
    return d'''
# %%
ord("!")


# %%
def bytes_to_unicode():
    """
    每个可能的字节（实际上是一个整数，范围为 0..255）会被 OpenAI 映射为一个可视化的 Unicode 字符。
    一些字节的外观会保持原样，因为它们没有引起任何问题，这些字节在 bs 列表中定义。例如：
    chr(33) 返回 "!"，因此在返回的字典中，我们直接有 d[33] -> "!"。
    然而，chr(0) 例如是 '\x00'，它看起来很难看。所以 OpenAI 将这些字节映射为新的字符，
    这些字符位于一个范围内，其中 chr() 返回一个漂亮的单个字符。
    所以在最终的字典中，我们将 d[0] -> 'Ā'，这实际上是 chr(0 + 2**8)。
    特别是，空格字符是 32，我们可以通过 ord(' ') 看到它。相反，这个函数将空格（32）偏移 256，
    得到 288， 所以 d[32] -> 'Ġ'。
    这个过程就是简单的将 0..255 范围内的字节映射到 Unicode 字符，一些字节保留原样，其他字节
    则映射到一个“好看”的字符，如 'Ā' 或 'Ġ' 等。
    """
    # 定义 188 个原样显示的字节，这些字节无需做任何转换
    bs = (
        list(range(ord("!"), ord("~") + 1))  # 从 '!' 到 '~' 的字符（ASCII 字符）
        + list(range(ord("¡"), ord("¬") + 1))  # 从 '¡' 到 '¬' 的字符（扩展拉丁字符）
        + list(range(ord("®"), ord("ÿ") + 1))  # 从 '®' 到 'ÿ' 的字符（其他扩展字符）
    )
    cs = bs[
        :
    ]  # 初始化 cs 为 bs 的副本，所有 bs 中的整数将直接映射到 chr(b) 在输出字典中

    # 处理其余 68 个需要转换的字节
    # 每个需要转换的字节将映射到 chr(256 + n)，n 从 0 增加到 67
    n = 0
    for b in range(2**8):  # 遍历所有可能的字节值 0 到 255
        if b not in bs:  # 如果字节 b 不在 bs 中（即是一个“丑陋”的字节）
            # 如果这个字节是“丑陋”的，则将其映射到下一个可用的“好看”字符
            bs.append(b)  # 将字节 b 添加到 bs 中
            cs.append(2**8 + n)  # 映射到一个新的 Unicode 字符（从 chr(256) 开始）
            n += 1  # 增加 n，分配下一个字符

    # 将 cs 中的整数值转换为对应的 Unicode 字符
    cs = [chr(n) for n in cs]

    # 创建一个字典，将字节值（bs）映射到对应的 Unicode 字符（cs）
    d = dict(zip(bs, cs))

    return d


def get_pairs(word):
    """
    Return all bigrams as a set of tuples, of consecutive elements in the iterable word.
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder, bpe_merges):
        # byte encoder/decoder
        # 初始化字节编码器和解码器
        self.byte_encoder = (
            bytes_to_unicode()
        )  # 获取一个映射字典，将整数映射到字符（即字节到Unicode的转换）
        self.byte_decoder = {
            v: k for k, v in self.byte_encoder.items()
        }  # 反转字节编码器的映射，得到解码器

        # bpe token encoder/decoder
        # BPE（Byte Pair Encoding）编码器和解码器
        self.encoder = encoder  # BPE编码器，通常是一个字典，将子词（或字符）映射到数字
        self.decoder = {
            v: k for k, v in self.encoder.items()
        }  # 反转BPE编码器的映射，得到解码器

        # bpe merge list that defines the bpe "tree", of tuples (a,b) that are to merge to token ab
        # BPE合并规则，定义了合并规则的"树"，即哪些字节对应该合并成一个新Token
        self.bpe_ranks = dict(
            zip(bpe_merges, range(len(bpe_merges)))
        )  # 将合并规则（例如('l', 'y')）与对应的优先级编号（范围从0到N）关联

        # the splitting pattern used for pre-tokenization
        # 用于预处理分词的正则表达式模式
        # 该模式在分词之前进行文本处理，处理包括对一些常见的缩写词（如Andrejs's）做特殊处理，并根据字母、数字、非字母数字、空格进行分词
        # 正则表达式的解释参考：
        """
        1. vertical bars | 是OR操作，re.findall会根据这些匹配项从左到右拆分文本。
        2. '\'s' 是用于匹配类似Andrejs's的字符串，将其拆分为 (Andrej, 's)。
        3. ' ?\p{L}+'：表示可选的空格后跟一个或多个Unicode字母字符。
        4. ' ?\p{N}+'：表示可选的空格后跟一个或多个Unicode数字字符。
        5. ' ?[^\s\p{L}\p{N}]+': 表示可选的空格后跟一个或多个不是字母、数字或空格的字符（即标点符号等）。
        6. '\s+(?!\S)': 表示匹配一个或多个空白字符，前提是后面没有非空白字符，这样可以匹配多个空格但不包括最后一个空格。
        7. '\s+': 表示匹配一个或多个空白字符，通常用来匹配字符串末尾的所有空白字符。

        总结：
        - 通过特殊处理常见的缩写（如's', 't', 're'等），将它们拆分为单独的Token。
        - 然后根据字母、数字、非字母数字和空格对文本进行分割。
        """
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        # cache用于存储缓存数据，避免重复计算
        self.cache = {}

    def bpe(self, token):
        """
        该函数使用 self.bpe_ranks 来迭代地合并所有可能的 BPE tokens，
        直到合并完成。token 是经过正则分词后的单个词（例如 'Ġthere'），并经过字节编码。
        """
        # token 是一个已经过字节编码的单个词（例如 'Ġthere'）

        # 记忆化优化，提高效率
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)  # 将token转换为元组，包含token中的每个字符
        pairs = get_pairs(word)  # 获取word中所有的字母对（bigrams）

        # 如果没有bigram，则不需要做任何合并，直接返回原始token
        if not pairs:
            return token

        while True:
            # 找到下一个最小的rank bigram，这个bigram是可以被合并的
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))

            # 如果bigram不在合并规则中，结束合并
            if bigram not in self.bpe_ranks:
                break  # 没有更多的bigram可以合并了

            first, second = bigram  # 提取bigram中的两个元素

            # 现在将当前列表中的所有(first, second)替换为合并后的Token 'first_second'
            new_word = []
            i = 0
            while i < len(word):
                # 找到当前词中first出现的下一个位置
                try:
                    j = word.index(first, i)  # 查找从索引i开始的first的下一个出现位置
                    new_word.extend(word[i:j])  # 将i到j之间的部分添加到new_word中
                    i = j  # 更新i为j，继续查找
                except:
                    # 如果没有找到，则将剩余部分添加到new_word中，结束循环
                    new_word.extend(word[i:])
                    break

                # 如果当前的first后面紧跟着second，则将它们合并成一个新Token
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)  # 合并first和second
                    i += 2  # 跳过这两个字符
                else:
                    new_word.append(word[i])  # 否则将当前字符添加到new_word中
                    i += 1  # 继续查找下一个字符

            # 所有(first, second)已经被合并为first_second
            new_word = tuple(new_word)  # 将新的词转换为元组
            word = new_word  # 更新word为新的词
            if len(word) == 1:
                break  # 如果词的长度为1，说明合并完成，退出循环
            else:
                pairs = get_pairs(word)  # 如果词的长度大于1，继续获取新的bigram进行合并

        # 将所有字符拼接成一个字符串，用空格作为分隔符。需要注意的是，
        # 现在所有字符都已经经过字节编码，确保空格不会在数据中出现，
        # 它是一个“特殊”的分隔符字符
        word = " ".join(word)

        # 将结果缓存并返回
        self.cache[token] = word
        return word

    def encode(self, text):
        """string goes in, list of integers comes out"""
        bpe_idx = []
        # pre-tokenize the input text into string tokens (words, roughly speaking)
        tokens = re.findall(self.pat, text)
        # process each token into BPE integers
        for token in tokens:
            # encode the token as a bytes (b'') object
            token_bytes = token.encode("utf-8")
            # translate all bytes to their unicode string representation and flatten
            token_translated = "".join(self.byte_encoder[b] for b in token_bytes)
            # perform all the applicable bpe merges according to self.bpe_ranks
            token_merged = self.bpe(token_translated).split(" ")
            # translate all bpe tokens to integers
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            # extend our running list of all output integers
            bpe_idx.extend(token_ix)
        return bpe_idx

    def encode_and_show_work(self, text):
        """debugging function, same as encode but returns all intermediate work"""
        bpe_idx = []
        parts = []
        tokens = re.findall(self.pat, text)
        for token in tokens:
            token_bytes = token.encode("utf-8")
            token_translated = "".join(self.byte_encoder[b] for b in token_bytes)
            token_merged = self.bpe(token_translated).split(" ")
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            bpe_idx.extend(token_ix)
            parts.append(
                {
                    "token": token,
                    "token_bytes": token_bytes,
                    "token_translated": token_translated,
                    "token_merged": token_merged,
                    "token_ix": token_ix,
                }
            )
        out = {
            "bpe_idx": bpe_idx,  # the actual output sequence
            "tokens": tokens,  # result of pre-tokenization
            "parts": parts,  # intermediates for each token part
        }
        return out

    def decode(self, bpe_idx):
        """list of integers comes in, string comes out"""
        # inverse map the integers to get the tokens
        tokens_merged = [self.decoder[token] for token in bpe_idx]
        # inverse the byte encoder, e.g. recovering 'Ġ' -> ' ', and get the bytes
        tokens_flat = "".join(tokens_merged)
        tokens_bytes = bytearray([self.byte_decoder[c] for c in tokens_flat])
        # recover the full utf-8 string
        text = tokens_bytes.decode("utf-8", errors="replace")
        return text


def get_file(local_file, remote_file):
    """downloads remote_file to local_file if necessary"""
    if not os.path.isfile(local_file):  # 如果本地文件不存在
        print(f"downloading {remote_file} to {local_file}")
        response = requests.get(remote_file)  # 从远程 URL 获取文件
        open(local_file, "wb").write(response.content)  # 保存文件到本地


def get_encoder():
    """
    Returns an instance of the GPT BPE Encoder/Decoder
    and handles caching of "database" files.
    """
    home_dir = os.path.expanduser("~")  # 获取用户的主目录
    cache_dir = os.path.join(home_dir, ".cache", "mingpt")  # 设置缓存路径
    os.makedirs(cache_dir, exist_ok=True)  # 如果不存在，创建该目录

    # load encoder.json that has the raw mappings from token -> bpe index下载 encoder.json 文件
    encoder_local_file = os.path.join(cache_dir, "encoder.json")
    encoder_remote_file = (
        "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json"
    )
    get_file(encoder_local_file, encoder_remote_file)
    with open(encoder_local_file, "r") as f:
        encoder = json.load(f)
    assert (
        len(encoder) == 50257  # 验证 encoder.json 中的条目数是否正确
    )  # 256 individual byte tokens, 50,000 merged tokens, and 1 special <|endoftext|> token

    # load vocab.bpe that contains the bpe merges, i.e. the bpe tree structure
    # in the form tuples (a, b), that indicate that (a, b) is to be merged to one token ab
    vocab_local_file = os.path.join(cache_dir, "vocab.bpe")
    vocab_remote_file = (
        "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe"
    )
    get_file(vocab_local_file, vocab_remote_file)
    with open(vocab_local_file, "r", encoding="utf-8") as f:
        bpe_data = f.read()
    # light postprocessing: strip the version on first line and the last line is a blank
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]
    assert len(bpe_merges) == 50000  # 50,000 merged tokens

    # construct the Encoder object and return
    enc = Encoder(encoder, bpe_merges)
    return enc


# -----------------------------------------------------------------------------


class BPETokenizer:
    """PyTorch-aware class that wraps the Encoder above"""

    def __init__(self):
        self.encoder = get_encoder()

    def __call__(self, text, return_tensors="pt"):
        # PyTorch only; here because we want to match huggingface/transformers interface
        assert return_tensors == "pt"
        # single string input for now, in the future potentially a list of strings
        assert isinstance(text, str)
        # encode and create a "batch dimension" of 1
        idx = [self.encoder.encode(text)]
        # wrap into PyTorch tensor
        out = torch.tensor(idx, dtype=torch.long)
        return out

    def decode(self, idx):
        # ensure a simple 1D tensor for now
        assert idx.ndim == 1
        # decode indices to text
        text = self.encoder.decode(idx.tolist())
        return text


if __name__ == "__main__":

    # here is an encoding example
    text = "Hello!! I'm Andrej Karpathy. It's 2022. w00t :D 🤗"
    e = get_encoder()
    r = e.encode_and_show_work(text)

    print("Original text is:")
    print(text)
    print("First the text gets pre-tokenized, broken up into chunks, the outcome is:")
    print(r["tokens"])
    # ['Hello', '!!', ' I', "'m", ' Andrej', ' Karpathy', '.', ' It', "'s", ' 2022', '.', ' w', '00', 't', ' :', 'D', ' 🤗']
    print("Then we iterate over each chunk and process them in turn...")
    for part in r["parts"]:
        print(part)
    # {'token': 'Hello', 'token_bytes': b'Hello', 'token_translated': 'Hello', 'token_merged': ['Hello'], 'token_ix': [15496]}
    # {'token': '!!', 'token_bytes': b'!!', 'token_translated': '!!', 'token_merged': ['!!'], 'token_ix': [3228]}
    # {'token': ' I', 'token_bytes': b' I', 'token_translated': 'ĠI', 'token_merged': ['ĠI'], 'token_ix': [314]}
    # {'token': "'m", 'token_bytes': b"'m", 'token_translated': "'m", 'token_merged': ["'m"], 'token_ix': [1101]}
    # {'token': ' Andrej', 'token_bytes': b' Andrej', 'token_translated': 'ĠAndrej', 'token_merged': ['ĠAndre', 'j'], 'token_ix': [10948, 73]}
    # {'token': ' Karpathy', 'token_bytes': b' Karpathy', 'token_translated': 'ĠKarpathy', 'token_merged': ['ĠK', 'arp', 'athy'], 'token_ix': [509, 5117, 10036]}
    # {'token': '.', 'token_bytes': b'.', 'token_translated': '.', 'token_merged': ['.'], 'token_ix': [13]}
    # {'token': ' It', 'token_bytes': b' It', 'token_translated': 'ĠIt', 'token_merged': ['ĠIt'], 'token_ix': [632]}
    # {'token': "'s", 'token_bytes': b"'s", 'token_translated': "'s", 'token_merged': ["'s"], 'token_ix': [338]}
    # {'token': ' 2022', 'token_bytes': b' 2022', 'token_translated': 'Ġ2022', 'token_merged': ['Ġ2022'], 'token_ix': [33160]}
    # {'token': '.', 'token_bytes': b'.', 'token_translated': '.', 'token_merged': ['.'], 'token_ix': [13]}
    # {'token': ' w', 'token_bytes': b' w', 'token_translated': 'Ġw', 'token_merged': ['Ġw'], 'token_ix': [266]}
    # {'token': '00', 'token_bytes': b'00', 'token_translated': '00', 'token_merged': ['00'], 'token_ix': [405]}
    # {'token': 't', 'token_bytes': b't', 'token_translated': 't', 'token_merged': ['t'], 'token_ix': [83]}
    # {'token': ' :', 'token_bytes': b' :', 'token_translated': 'Ġ:', 'token_merged': ['Ġ:'], 'token_ix': [1058]}
    # {'token': 'D', 'token_bytes': b'D', 'token_translated': 'D', 'token_merged': ['D'], 'token_ix': [35]}
    # {'token': ' 🤗', 'token_bytes': b' \xf0\x9f\xa4\x97', 'token_translated': 'ĠðŁ¤Ĺ', 'token_merged': ['ĠðŁ', '¤', 'Ĺ'], 'token_ix': [12520, 97, 245]}
    # (refer to the code inside Encoder.encode for what these intermediates are)
    print("and the final outcome is concatenating and flattening all the token_ix:")
    print(r["bpe_idx"])
    # [15496, 3228, 314, 1101, 10948, 73, 509, 5117, 10036, 13, 632, 338, 33160, 13, 266, 405, 83, 1058, 35, 12520, 97, 245]
    # this would then become the integer input sequence to the transformer
    print("ready to feed into a Transformer!")
