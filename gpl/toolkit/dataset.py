from typing import Dict
import linecache
from datasets import Dataset

def concat_title_and_body(did: str, corpus: Dict[str, Dict[str, str]], sep: str):
    assert type(did) == str
    document = []
    title = corpus[did]["title"].strip()
    body = corpus[did]["text"].strip()
    if len(title):
        document.append(title)
    if len(body):
        document.append(body)
    return sep.join(document)

class HardNegativeDataset(Dataset):
    def __init__(self, jsonl_path, queries, corpus, sep=" "):
        self.jsonl_path = jsonl_path
        self.queries = queries
        self.corpus = corpus
        self.sep = sep
        self.none_indices = set()
        self.nqueries = len(linecache.getlines(jsonl_path))

    def __getitem__(self, item):
        shift = 0
        while True:
            index = (item + shift) % self.nqueries + 1
            shift += 1
            if index in self.none_indices:
                continue
            json_line = linecache.getline(self.jsonl_path, index)
            try:
                query_dict = json.loads(json_line)
            except:
                print(json_line, "###index###", index)
                raise NotImplementedError
            tuple_sampled = self._sample_tuple(query_dict)
            if tuple_sampled is None:
                self.none_indices.add(index)
                logger.info(f"Invalid query at line {index-1}")
            else:
                break
        (query_id, pos_id, neg_id), (query_text, pos_text, neg_text) = tuple_sampled
        return InputExample(
            guid=[query_id, pos_id, neg_id],
            texts=[query_text, pos_text, neg_text],
            label=-1,
        )

    def __len__(self):
        return self.nqueries

    def _sample_tuple(self, query_dict):
        # Get the positive passage ids
        pos_pids = query_dict["pos"]
        # scores = {item['pid']: item['ce-score'] for item in query_dict['pos']}

        # Get the hard negatives
        neg_pids = set()
        for system_name, system_negs in query_dict["neg"].items():
            for pid in system_negs:

                if pid not in neg_pids:
                    neg_pids.add(pid)
                    # scores[pid] = item['ce-score']

        if len(pos_pids) > 0 and len(neg_pids) > 0:
            query_text = self.queries[query_dict["qid"]]

            pos_pid = random.choice(pos_pids)
            pos_text = concat_title_and_body(pos_pid, self.corpus, self.sep)

            neg_pid = random.choice(list(neg_pids))
            neg_text = concat_title_and_body(neg_pid, self.corpus, self.sep)

            return (query_dict["qid"], pos_pid, neg_pid), (
                query_text,
                pos_text,
                neg_text,
            )
        else:
            return None


def build_hf_dataset(tsv_path, queries, corpus, sep=" ", max_lines=None):
    query_texts = []
    pos_texts = []
    neg_texts = []
    labels = []

    with open(tsv_path, "r") as f:
        total_lines = sum(1 for _ in f)

    if max_lines:
        total_lines = min(max_lines, total_lines)

    for idx in range(total_lines):
        line = linecache.getline(tsv_path, idx + 1)  # linecache is 1-indexed
        if not line.strip():
            continue
        try:
            qid, pos_pid, neg_pid, label = line.strip().split("\t")
        except ValueError:
            continue  # malformed line

        query_texts.append(queries[qid])
        pos_texts.append(concat_title_and_body(pos_pid, corpus, sep))
        neg_texts.append(concat_title_and_body(neg_pid, corpus, sep))
        labels.append(float(label))

    hf_dataset = Dataset.from_dict({
        "text1": query_texts,
        "text2": pos_texts,
        "text3": neg_texts,
        "label": labels
    })

    return hf_dataset