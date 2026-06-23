#〔讲解〕本文件属于 scripts/(数据准备)环节,是整条 pipeline 的「第 0 步」。
#〔讲解〕作用:把 AggAgent 评测要用的几个搜索问答 benchmark(BrowseComp / DeepSearchQA /
#〔讲解〕HealthBench-hard / ResearchRubrics / BrowseComp+)从网上下载下来,统一转成本地的
#〔讲解〕`data/*.jsonl`(每行一条 {problem, answer, ...})。
#〔讲解〕下游:rollout/ 阶段读这些 jsonl 里的 problem 去跑 ReAct 搜索轨迹,evaluation/ 阶段拿 answer 判分。
#〔讲解〕特别地:BrowseComp 的题面/答案是加密存放的(防止被搜索引擎/模型预训练偷看到答案),本文件负责解密。
import argparse
import base64
import hashlib
import json
import os
import shutil
import urllib.request

import pandas


#〔讲解〕所有数据集统一落盘到当前工作目录下的 data/ 文件夹。
DATA_DIR = "data"


#〔讲解〕derive_key:由一个口令(password)派生出指定长度(length)的字节密钥,供 XOR 解密用。
#〔讲解〕入参 password=口令字符串(这里就是每行的 canary 值);length=需要的密钥总字节数(=密文长度)。
#〔讲解〕返回:长度恰为 length 的 bytes 密钥。
def derive_key(password: str, length: int) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    #〔讲解〕先把口令做 SHA-256,得到固定 32 字节的摘要 key;但密文可能比 32 字节长,需要把 key 重复铺满。
    key = hasher.digest()
    #〔讲解〕核心:把 32 字节的 key 重复(length//32)整段,再补上余数那几个字节,拼成恰好 length 长的密钥流。
    #〔讲解〕这是一种最朴素的「重复密钥流」做法(类似维吉尼亚/流密码),只为隐藏明文、非真正强加密。
    return key * (length // len(key)) + key[: length % len(key)]


#〔讲解〕decrypt:把 base64 编码的密文,用 password 派生的密钥做逐字节 XOR 解密,还原成明文字符串。
#〔讲解〕这是 BrowseComp 官方约定的解密法:题面/答案先 XOR 加密再 base64,canary 当解密口令。
def decrypt(ciphertext_b64: str, password: str) -> str:
    #〔讲解〕第一步:base64 解码,拿回原始的密文字节。
    encrypted = base64.b64decode(ciphertext_b64)
    #〔讲解〕第二步:按密文长度派生等长的密钥流。
    key = derive_key(password, len(encrypted))
    #〔讲解〕第三步:密文与密钥逐字节 XOR(异或),即得明文字节。XOR 的特性是「加密=解密同一操作」。
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


#〔讲解〕download_browsecomp:下载 OpenAI 的 BrowseComp 测试集(加密 CSV),解密后存成 browsecomp.jsonl。
#〔讲解〕这是本脚本的「主数据集」——无论命令行加不加参数,main 里都会先跑它。
def download_browsecomp():
    print("Downloading BrowseComp dataset...")
    #〔讲解〕直接用 pandas 从 OpenAI 公开 blob 地址读取 CSV(含 canary/problem/answer 列,后两者是密文)。
    df = pandas.read_csv(
        "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "browsecomp.jsonl")

    #〔讲解〕幂等保护:若目标文件已存在就跳过,避免重复下载/解密(本脚本所有数据集都遵循这个模式)。
    if os.path.exists(out_path):
        print(f"Already exists, skipping: {out_path}")
        return

    with open(out_path, "w") as f:
        #〔讲解〕逐行遍历 CSV;每行都有自己的 canary 当解密口令。
        for _, row in df.iterrows():
            canary = row["canary"]
            #〔讲解〕用本行 canary 把密文的 problem(题面)解密成明文。
            problem = decrypt(row["problem"], canary)
            #〔讲解〕同理解密 answer(标准答案)。
            answer = decrypt(row["answer"], canary)
            record = {"problem": problem, "answer": answer}
            #〔讲解〕jsonl 格式:一行一个 JSON 对象,末尾补换行;这是下游 rollout/evaluation 统一的输入格式。
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(df)} examples to {out_path}")


#〔讲解〕download_deepsearchqa:下载 Google 的 DeepSearchQA 数据集(走 HuggingFace datasets),取 eval 划分存盘。
#〔讲解〕与 BrowseComp 不同,这个数据集是明文的,无需解密;多带了题目分类和答案类型两个元字段。
def download_deepsearchqa():
    print("Downloading DeepSearchQA dataset...")
    out_path = os.path.join(DATA_DIR, "deepsearchqa.jsonl")

    if os.path.exists(out_path):
        print(f"Already exists, skipping: {out_path}")
        return

    #〔讲解〕局部 import:datasets 是 HuggingFace 的重型库,放函数内部按需加载,没下这个集时不必装它。
    from datasets import load_dataset

    #〔讲解〕从 HF Hub 拉取 google/deepsearchqa 仓库的 "deepsearchqa" 配置(config)。
    dataset = load_dataset("google/deepsearchqa", "deepsearchqa")
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(out_path, "w") as f:
        #〔讲解〕只取 "eval" 这个 split(评测集);逐条抽出需要的字段写成 jsonl。
        for example in dataset["eval"]:
            record = {
                "problem": example["problem"],
                "answer": example["answer"],
                #〔讲解〕problem_category=题目类别,answer_type=答案类型(如数值/实体),供后续分桶统计或判分用。
                "problem_category": example["problem_category"],
                "answer_type": example["answer_type"],
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(dataset['eval'])} examples to {out_path}")


#〔讲解〕download_healthbench:下载 OpenAI HealthBench 的 hard 子集(已是 jsonl 文件),直接整文件搬运。
#〔讲解〕这个最简单——源就是 jsonl,不用逐行解析转换,urlretrieve 把整个文件抓下来即可。
def download_healthbench():
    print("Downloading HealthBench (hard) dataset...")
    out_path = os.path.join(DATA_DIR, "healthbench.jsonl")

    if os.path.exists(out_path):
        print(f"Already exists, skipping: {out_path}")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    url = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl"
    #〔讲解〕urlretrieve:把 URL 指向的文件原样下载落地到 out_path(适合源本身就是目标格式的情况)。
    urllib.request.urlretrieve(url, out_path)

    #〔讲解〕重新打开数下载到的行数,只为打印「存了多少条」给用户确认。
    with open(out_path) as f:
        count = sum(1 for _ in f)
    print(f"Saved {count} examples to {out_path}")


#〔讲解〕download_researchrubrics:下载 ScaleAI 的 ResearchRubrics 数据集(HF datasets),取 train 划分存盘。
#〔讲解〕这个集是「带评分细则(rubrics)」的开放式研究题,字段比前面多——评判靠 rubrics 而非单一标准答案。
def download_researchrubrics():
    print("Downloading ResearchRubrics dataset...")
    out_path = os.path.join(DATA_DIR, "researchrubrics.jsonl")

    if os.path.exists(out_path):
        print(f"Already exists, skipping: {out_path}")
        return

    from datasets import load_dataset

    dataset = load_dataset("ScaleAI/researchrubrics")
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(out_path, "w") as f:
        #〔讲解〕该数据集只有 train 一个 split,逐条抽字段写盘。
        for example in dataset["train"]:
            record = {
                "prompt": example["prompt"],
                "sample_id": example["sample_id"],
                #〔讲解〕domain=领域;后面三项(概念广度/逻辑嵌套深度/探索性)是该题难度的多维标注。
                "domain": example["domain"],
                "conceptual_breadth": example["conceptual_breadth"],
                "logical_nesting": example["logical_nesting"],
                "exploration": example["exploration"],
                #〔讲解〕rubrics=评分细则清单(LLM judge 据此逐条打分),是开放式题型评测的核心字段。
                "rubrics": example["rubrics"],
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(dataset['train'])} examples to {out_path}")


#〔讲解〕download_browsecomp_plus:准备 BrowseComp+ 这套「带固定语料库 + 检索索引」的离线检索评测。
#〔讲解〕区别于普通 BrowseComp(联网搜),BrowseComp+ 让 agent 在一个固定 corpus 上检索,需额外下载语料和向量索引。
#〔讲解〕只有命令行显式传 --browsecomp-plus 时才会走这条分支。
def download_browsecomp_plus():
    #〔讲解〕局部 import:snapshot_download 用来整仓库批量下载 HF 上的数据集文件。
    from huggingface_hub import snapshot_download

    # Create browsecomp-plus.jsonl as a copy of browsecomp.jsonl
    #〔讲解〕BrowseComp+ 的题面/答案与普通 BrowseComp 相同,所以直接复用——把 browsecomp.jsonl 拷一份改名即可。
    src_path = os.path.join(DATA_DIR, "browsecomp.jsonl")
    out_path = os.path.join(DATA_DIR, "browsecomp-plus.jsonl")
    if os.path.exists(out_path):
        print(f"Already exists, skipping: {out_path}")
    else:
        #〔讲解〕兜底:若还没下过普通 BrowseComp,先递归调用把它下下来,保证 src_path 存在。
        if not os.path.exists(src_path):
            download_browsecomp()
        shutil.copy(src_path, out_path)
        print(f"Copied {src_path} -> {out_path}")

    browsecomp_plus_dir = os.path.join(DATA_DIR, "browsecomp-plus")
    os.makedirs(browsecomp_plus_dir, exist_ok=True)

    # Download retrieval indexes (qwen3-embedding-8b)
    #〔讲解〕检索索引:预先用 qwen3-embedding-8b 模型把语料编码好的向量索引,agent 检索时直接拿来用,省去自己建库。
    indexes_dir = os.path.join(browsecomp_plus_dir, "indexes")
    if os.path.exists(indexes_dir):
        print(f"Indexes already exist, skipping: {indexes_dir}")
    else:
        print("Downloading BrowseComp+ indexes...")
        snapshot_download(
            repo_id="Tevatron/browsecomp-plus-indexes",
            repo_type="dataset",
            #〔讲解〕allow_patterns:只下载该子目录,这个大仓库里有多种 embedding 模型的索引,此处只要 qwen3-embedding-8b 这一份。
            allow_patterns="qwen3-embedding-8b/*",
            local_dir=indexes_dir,
        )
        print(f"Saved indexes to {indexes_dir}")

    # Download corpus (kept as parquet)
    #〔讲解〕语料库:被检索的全部文档(网页正文等),保持 parquet 列式格式存放,供检索器读取并返回命中文档。
    corpus_dir = os.path.join(browsecomp_plus_dir, "corpus")
    if os.path.exists(corpus_dir):
        print(f"Corpus already exists, skipping: {corpus_dir}")
    else:
        print("Downloading BrowseComp+ corpus...")
        snapshot_download(
            repo_id="Tevatron/browsecomp-plus-corpus",
            repo_type="dataset",
            local_dir=corpus_dir,
        )
        print(f"Saved corpus to {corpus_dir}")


#〔讲解〕命令行入口:决定本次下载哪些数据集。两种模式互斥(见下)。
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #〔讲解〕唯一开关 --browsecomp-plus:store_true 表示「带上这个 flag 即为 True,不带为 False」。
    parser.add_argument("--browsecomp-plus", action="store_true",
                        help="Also download BrowseComp+ indexes and corpus")
    args = parser.parse_args()

    #〔讲解〕无论哪种模式,都先下主数据集 BrowseComp。
    download_browsecomp()
    #〔讲解〕分支逻辑:加了 --browsecomp-plus → 只补下 BrowseComp+(语料+索引);
    #〔讲解〕否则 → 下另外三个常规数据集(DeepSearchQA / HealthBench / ResearchRubrics)。二者不同时进行。
    if args.browsecomp_plus:
        download_browsecomp_plus()
    else:
        download_deepsearchqa()
        download_healthbench()
        download_researchrubrics()


