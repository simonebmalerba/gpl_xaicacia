import os
import json
import textwrap

def beir2humanread(query_file, corpus_file, tsv_file, output_file, sep=" ",width=150):
    '''
        Convert BEIR style qa dataset to human readable format.
    '''
    queries = {}
    with open(query_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            queries[item["_id"]] = item["text"]

    # Load corpus
    corpus = {}
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            corpus[item["_id"]] = sep.join([item.get("title", ""), item.get("text", "")]).strip()

    # Read triplets and extract QA pairs
    with open(tsv_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            for line in f_in:
                parts = line.strip().split("\t")

                if len(parts) < 2:
                    continue  # skip malformed lines

                qid, pos_pid = parts[0], parts[1]
                neg_id = parts[2] if len(parts) > 2 else None
                margin = parts[3] if len(parts) > 3 else None

                question = queries.get(qid, "").strip()
                answer = corpus.get(pos_pid, "").strip()
                negative = corpus.get(neg_id, "").strip() if neg_id else None

                if question and answer:
                    f_out.write("question:\n" + textwrap.fill(question, width) + "\n")
                    f_out.write("answer:\n" + textwrap.fill(answer, width) + "\n")
                    if negative:
                        f_out.write("negative:\n" + textwrap.fill(negative, width) + "\n")
                    if margin:
                        f_out.write(f"Label distance: {margin}\n")
                    f_out.write("---\n")
                    
                    
def st2humanread(json_file, output_file, sep=" ",width=150):
    '''
        Convert ST style qa dataset to human readable format.
    '''

    # Read triplets and extract QA pairs
    with open(json_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        with open(json_file) as f:
            qa_data = json.load(f)
        for qid,query in qa_data["queries"].items():
            relevant_chunks_ids = qa_data["relevant_docs"].get(qid, [])
            for chunk_id in relevant_chunks_ids:
                chunk = qa_data["corpus"].get(chunk_id, "").strip()
                if query and chunk:
                    f_out.write("question:\n" + textwrap.fill(query, width) + "\n")
                    f_out.write("answer:\n" + textwrap.fill(chunk, width) + "\n")
                    f_out.write("---\n")
                
def qa_negatives_tobgem3(query_file, corpus_file, hn_file, output_file, sep=" ",width=150,reranker = 'msmarco-distilbert-base-v3'):
    '''
        Convert BEIR style qa dataset to BGE-m3 like (jsonl with dictionaries)
    '''
    
    queries = {}
    with open(query_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            queries[item["_id"]] = item["text"]

    # Load corpus
    corpus = {}
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            corpus[item["_id"]] = sep.join([item.get("title", ""), item.get("text", "")]).strip()
    with open(hn_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            item = json.loads(line)
            qid = item["qid"]
            query_text = queries.get(qid)
            if not query_text:
                pass
            positives = [corpus[pid] for pid in item.get('pos', []) if pid in corpus]
            negatives = [corpus[nid] for nid in item.get('neg', [])[reranker] if nid in corpus]
            out = {
                "query": query_text,
                "pos": positives,
                "neg": negatives,
                "pos_scores": [],
                "neg_scores": [],
                "prompt": "query:",
                "type": "normal"
            }
            f_out.write(json.dumps(out) + "\n")