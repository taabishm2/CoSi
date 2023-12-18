import io
import os
import re
import ast
import csv
import json
import time
import glob
import pickle
import datetime
import tokenize
import threading
import numpy as np
import pandas as pd

from math import e
from pprint import pprint
from dotenv import load_dotenv
from collections import Counter
from difflib import SequenceMatcher
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


load_dotenv("./files/.env")

SCORES_LOCK = threading.Lock()
SEMAPHORE = threading.Semaphore(int(os.getenv('MAX_PARALLEL_THREADS')))

BASE_DIR = "./files"
FINGERPRINT_DIR = BASE_DIR + "/fingerprint_cache/"
SUBMISSION_DIR = BASE_DIR + "/submissions/"

DEBUG_MODE = bool(int(os.getenv('DEBUG_MODE')))

HASH_WINDOW_SIZE = int(os.getenv('FINGERPRINT_WINDOW_SIZE'))
CACHE_FINGERPRINTS = bool(int(os.getenv('CACHE_FINGERPRINTS')))
SESSION_ID = os.getenv('FINGERPRINT_SESSION_ID')
USE_FINGERPRINT_CACHE = bool(int(os.getenv('USE_FINGERPRINT_CACHE')))
INCLUDE_FINGERPRINTS_FROM = FINGERPRINT_DIR + os.getenv('INCLUDE_FINGERPRINTS_IN')
SAVE_FINGERPRINTS_TO = FINGERPRINT_DIR + os.getenv('SAVE_FINGERPRINTS_IN')

IGNORE_PARSE_FAILURES = bool(int(os.getenv('IGNORE_PARSE_FAILURES')))

SUBMISSIONS_ROOT_DIR = SUBMISSION_DIR + os.getenv("SUBMISSIONS_ROOT_DIR")
FILES_IN_SUBMISSION = sorted(os.getenv("FILES_IN_SUBMISSION").split(","))
GITHUB_PREFIX = os.getenv('GITHUB_PREFIX')

IGNORED_TOKEN_NAMES = {
    "COMMENT",
    "ENDMARKER",
    "NEWLINE",
    "INDENT",
    "DEDENT",
    "NL",
    "ENCODING",
    "TYPE_COMMENT",
    "TYPE_IGNORE",
}

FINAL_RESULT = []


def replace_char_at_index(s, i, c):
    return s[:i] + c + s[i + 1 :]


def find_positions(text, pattern, tgt_char):
    positions = []
    for match in re.finditer(pattern, text):
        start_pos = match.start() + match.group().index(tgt_char)
        positions.append(start_pos)
    return positions


def get_variable_names(py_code):
    tree = ast.parse(py_code)
    variables = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    variables.add(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for elem in target.elts:
                        if isinstance(elem, ast.Name):
                            variables.add(elem.id)
                elif isinstance(target, ast.Attribute): 
                    if isinstance(target.value, ast.Name):
                        variables.add(target.attr)
        elif isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name):
                    variables.add(base.id)
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.Lambda):
            for arg in node.args.args:
                variables.add(arg.arg)
        elif isinstance(node, ast.Tuple):
            for elem in node.elts:
                if isinstance(elem, ast.Starred):
                    variables.add(elem.value.id)
        elif isinstance(node, ast.Global) or isinstance(node, ast.Nonlocal):
            for name in node.names:
                variables.add(name)
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                variables.add(kw.arg)
        elif isinstance(node, ast.Import):
            for name in node.names:
                if name.asname:
                    variables.add(name.asname)
        elif isinstance(node, ast.ImportFrom):
            for n in node.names:
                if n.asname:
                    variables.add(n.asname)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                variables.add(node.name)
        elif isinstance(node, ast.For):
            if isinstance(node.target, ast.Name):
                variables.add(node.target.id)
            elif isinstance(node.target, (ast.Tuple, ast.List)):
                for elem in node.target.elts:
                    if isinstance(elem, ast.Name):
                        variables.add(elem.id)
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                variables.add(node.target.id)
        elif isinstance(node, ast.With):
            for item in node.items:
                if isinstance(item.context_expr, ast.Name):
                    variables.add(item.context_expr.id)
        elif isinstance(node, ast.FunctionDef):
            variables.add(node.name)
        elif isinstance(
            node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        ):
            for comp in node.generators:
                if isinstance(comp.target, ast.Name):
                    variables.add(comp.target.id)
    
    return [v for v in variables if v is not None]


def get_submission_paths(path_pattern):
    return sorted(glob.glob(path_pattern))


def get_code_and_outputs(file_path):
    if not file_path.endswith(".py") and not file_path.endswith(".ipynb"):
        raise Exception(f"Can only check .py and .ipynb files, not {file_path}")

    if file_path.endswith(".py"):
        with open(file_path, "r") as f:
            return f.read(), "PLACEHOLDER_OUTPUT"

    with open(file_path) as f:
        nb = json.load(f)
        if "cells" not in nb:
            return []

        py_cells, output_cells = [], []
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue

            cell_source = "".join(cell["source"]).strip()
            for output in cell.get("outputs", []):
                if "text" in output:
                    output_cells.append("".join(output["text"]))
                elif "text/plain" in output.get("data", {}):
                    output_cells.append("".join(output["data"]["text/plain"]))

            if cell_source.startswith("!"):
                py_cells.append("# " + cell_source)
            py_cells.append(cell_source)

        py_code, py_out = ("\n".join(py_cells), "\n".join(output_cells))
        
        # Comment out bash commands 
        bash_positions = find_positions(py_code, r'((?<=\n)|(?<=^))(\!).*', "!")
        for p in bash_positions:
            py_code = replace_char_at_index(py_code, p, "#")
        
        bash_positions = find_positions(py_code, r'((?<=\n)|(?<=^))(%).*', "%")
        for p in bash_positions:
            py_code = replace_char_at_index(py_code, p, "#")

        return py_code, py_out


def get_token_list(py_code, var_names={}):
    token_list = []
    for token in tokenize.generate_tokens(io.StringIO(py_code).readline):
        tok_type, tok_str = tokenize.tok_name[token.type], token.string

        if tok_type in IGNORED_TOKEN_NAMES:
            continue
        if tok_type == "NAME" and tok_str in var_names:
            tok_str = "@"
        if tok_type == "STRING" and 'f"' in tok_str:
            tok_str = re.sub(r"\{.*?\}", "@", tok_str)

        token_list.append(tok_str)

    return token_list

def get_score(x):
    if x < 1: return 0
    if x < 4: return 1
    if x > 50: return 0.1 
    return -0.019149*x + 1.057446809


HASH_WINDOW_DICT = dict()
def get_winnowed_hash_counts(token_list_dict):
    hash_set_dict, hash_to_source = defaultdict(dict), defaultdict(set)
    for path in token_list_dict:
        if not path in hash_set_dict:
            hash_set_dict[path] = defaultdict(int)
        for i in range(len(token_list_dict[path]) - HASH_WINDOW_SIZE):
            window_hash = hash("".join(token_list_dict[path][i : i + HASH_WINDOW_SIZE]))
            # For debugging only
            # HASH_WINDOW_DICT[window_hash] = "".join(token_list_dict[path][i : i + HASH_WINDOW_SIZE])
            # print(path, window_hash)
            hash_set_dict[path][window_hash] += 1
            hash_to_source[window_hash].add(path)

    winnowed_hash_counts = dict()
    for path in hash_set_dict:
        if path not in winnowed_hash_counts:
            winnowed_hash_counts[path] = defaultdict(int)
        for digest in hash_set_dict[path]:            
            digest_score = get_score(len(hash_to_source[digest]) - 1)
            winnowed_hash_counts[path][digest] += digest_score

    return winnowed_hash_counts


def get_sim_vector(path1, path2, full_vector_dict, var_dict, out_dict, src_code_dict):
    code_sim = cosine_similarity([full_vector_dict[path1]], [full_vector_dict[path2]])[0][0]
    var_sim = get_vars_similarity(var_dict[path1], var_dict[path2])
    output_sim = calculate_text_similarity(out_dict[path1], out_dict[path2])
    structure_sim = calculate_structure_similarity(src_code_dict[path1], src_code_dict[path2])

    precision_digits = 4
    return [
            round(code_sim, precision_digits),
            round(var_sim, precision_digits),
            round(output_sim, precision_digits),
            round(structure_sim, precision_digits)
        ]


def worker(path1, path2, full_vector_dict, var_dict, out_dict, src_code_dict, p1,p2):
    path1_stub = path1.split("/")[-1]
    path1_url = GITHUB_PREFIX + path1_stub

    path2_stub = path2.split("/")[-1]
    path2_url = GITHUB_PREFIX + path2_stub
    
    scores = [path1_url, path2_url] + \
        get_sim_vector(path1, path2, full_vector_dict, var_dict, out_dict, src_code_dict) + \
        [f"python3 display.py '{path1_stub}' '{path2_stub}'"]
        
    with SCORES_LOCK:
        FINAL_RESULT.append(scores)
    # print(f"Compared {p1} with {p2}")


def get_similarity_scores(paths, winnowed_hash_counts, var_dict, out_dict, src_code_dict):
    all_keywords = set()
    for path in winnowed_hash_counts:
        all_keywords = all_keywords.union(winnowed_hash_counts[path].keys())
    all_keywords = sorted(list(all_keywords))

    full_vector_dict = {
        path: np.array(
            [winnowed_hash_counts[path].get(keyword, 0) for keyword in all_keywords]
        )
        for path in paths
    }

    threads = []
    for p1 in range(len(paths) - 1):
        t1 = time.time()
        if DEBUG_MODE: print(f"Comparing: {p1}/{len(paths) - 1}", end="")
        for p2 in range(p1 + 1, len(paths)):
            path1, path2 = paths[p1], paths[p2]

            # Uncomment to see differences
            # if DEBUG_MODE:
            #     print("\nDifferences:")
            #     for k in all_keywords:
            #         # print(k, k in winnowed_hash_counts[path1], k in winnowed_hash_counts[path1])
            #         if (k in winnowed_hash_counts[path1] and k not in winnowed_hash_counts[path2]) or (k not in winnowed_hash_counts[path1] and k in winnowed_hash_counts[path2]):
            #             print(str.ljust(HASH_WINDOW_DICT[k], 30), f"{path1}:{k in winnowed_hash_counts[path1]}, {path2}:{k in winnowed_hash_counts[path2]}")
            
            t = threading.Thread(target=worker, args=(path1, path2, full_vector_dict, var_dict, out_dict, src_code_dict, p1,p2))
            with SEMAPHORE:
                t.start()
                threads.append(t)
        print(f", took {time.time()-t1}s")
    
    # Wait for all threads to complete
    for t in threads:
        t.join()


def calculate_text_similarity(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()

    return cosine_similarity(vectors)[0, 1]


def levenshtein_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()


def jaccard_similarity(list1, list2):
    intersection = len(set(list1).intersection(set(list2)))
    union = len(set(list1)) + len(set(list2)) - intersection
    return intersection / union


def find_best_match(name, name_list):
    max_similarity = 0
    best_match = None

    for candidate in name_list:
        similarity = levenshtein_ratio(name, candidate)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = candidate

    return best_match, max_similarity


def get_vars_similarity(names1, names2):
    total_similarity = 0
    matched_names = set()

    for name in names1:
        best_match, similarity = find_best_match(name, names2)
        if best_match and best_match not in matched_names:
            matched_names.add(best_match)
            total_similarity += similarity

    avg_similarity = total_similarity / len(names1) if names1 else 0
    jaccard_sim = jaccard_similarity(names1, names2)

    return 0.5 * avg_similarity + 0.5 * jaccard_sim


def calculate_structure_similarity(src1, src2):
    structure1 = construct_structure(ast.parse(src1))
    structure2 = construct_structure(ast.parse(src2))

    return jaccard_similarity(structure1, structure2)


def construct_structure(node, level=0):
    structure = []
    structure.append(type(node).__name__)
    for child in ast.iter_child_nodes(node):
        structure.extend(construct_structure(child, level+1))
    return structure


def main():
    t1 = time.time()
    token_list_dict = defaultdict(list)
    src_code_dict = dict()
    variable_names_dict = dict()
    output_txt_dict = dict()
    failed_files = dict()

    all_repos = []
    for repo in next(os.walk(SUBMISSIONS_ROOT_DIR))[1]:
        repo = os.path.join(SUBMISSIONS_ROOT_DIR, repo)
        if repo == SUBMISSIONS_ROOT_DIR: continue

        combined_submission_content = []
        try:
            for file_pattern in FILES_IN_SUBMISSION:
                py_code, output_text = get_code_and_outputs(f"{repo}/{file_pattern}")
                var_names = get_variable_names(py_code)
                tokens = get_token_list(py_code, var_names)

                combined_submission_content.append((tokens, output_text, var_names, py_code))
        except Exception as e:
            print(f"[FAILURE] for {repo}, {e}")
            if not IGNORE_PARSE_FAILURES: raise e 
            failed_files[repo] = str(e)

        if repo in failed_files: continue
        all_repos.append(repo)

        all_vars, all_outputs, all_tokens, src_code = set(), "", [], ""
        for t, o, v, c in combined_submission_content: 
            all_vars.update(v)
            all_outputs += o
            all_tokens.extend(t)
            src_code = c

        variable_names_dict[repo] = all_vars
        output_txt_dict[repo] = all_outputs
        token_list_dict[repo] = all_tokens
        src_code_dict[repo] = src_code

    winnowed_hash_counts = get_winnowed_hash_counts(token_list_dict)

    if CACHE_FINGERPRINTS:
        file_path = SAVE_FINGERPRINTS_TO + "/" + SESSION_ID + ".pkl"
        if not os.path.exists(SAVE_FINGERPRINTS_TO): os.makedirs(SAVE_FINGERPRINTS_TO)
        with open(file_path, 'wb') as f:
            if DEBUG_MODE: print(f"Saving fingerprints in {file_path}")
            pickle.dump(winnowed_hash_counts, f)

    if DEBUG_MODE: print(f"Current submissions to check: {len(winnowed_hash_counts)}")

    if USE_FINGERPRINT_CACHE:
        if not os.path.exists(INCLUDE_FINGERPRINTS_FROM):
            print("[ERROR] Fingerprint directory not found. Ignoring USE_FINGERPRINT_CACHE...")
        
        if DEBUG_MODE: print(f"Using fingerprints in {INCLUDE_FINGERPRINTS_FROM}")
        for root, _, filenames in os.walk(INCLUDE_FINGERPRINTS_FROM):
            for filename in filenames:
                if filename.endswith(".pkl"):
                    full_path = os.path.join(root, filename)
                    with open(full_path, "rb") as f:
                        prev_fingerprints = pickle.load(f)
                        winnowed_hash_counts.update(prev_fingerprints)

    if DEBUG_MODE: print(f"Total submissions to check: {len(winnowed_hash_counts)}")

    get_similarity_scores(all_repos, winnowed_hash_counts, variable_names_dict, output_txt_dict, src_code_dict)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"./files/results/{current_time}.csv", "w") as res_file:
        csv_w = csv.writer(res_file)
        csv_w.writerow(["Repo A", "Repo B", "Code", "Variables", "Output", "Structure", "Visualize"])
        csv_w.writerows(FINAL_RESULT)
    
    if len(failed_files) > 0:
        with open(f"./files/results/FAILED-{current_time}.csv", "w") as res_file:
            csv_w = csv.writer(res_file)
            csv_w.writerow(["Path", "Error"])
            csv_w.writerows([[i, failed_files[i]] for i in failed_files])

    print(f"\nChecked {len(all_repos)} files in {(time.time()-t1)//60} mins with {len(failed_files)} failures")

if __name__ == '__main__':
    main()



