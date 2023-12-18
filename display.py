import os
import sys
import requests
import webbrowser
import random
import tempfile
import argparse
import os
import subprocess

from CoSi import get_code_and_outputs, SUBMISSIONS_ROOT_DIR
from dotenv import load_dotenv

load_dotenv("./files/.env")

FILES_IN_SUBMISSION = sorted(os.getenv("FILES_IN_SUBMISSION").split(","))
GITHUB_PREFIX = os.getenv('GITHUB_PREFIX')


def generate_diff(text1, text2, email):
    url = 'https://api.diffchecker.com/public/text'
    params = {'output_type': 'html', 'email': email}
    payload = {
        'left': text1,
        'right': text2,
        'diff_level': 'character'
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, params=params, json=payload, headers=headers)
    
    if response.status_code == 200:
        html_content = response.text
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
            f.write(html_content)
        webbrowser.open_new_tab(f.name)
    else:
        raise Exception("API call failed." + str(response))


def get_code(repo_path):
    full_code = [f"{'*'*50}\n# Submission: {repo_path}\n{'*'*50}\n"]
    for file_pattern in FILES_IN_SUBMISSION:
        code_str, _ = get_code_and_outputs(f"{repo_path}/{file_pattern}")
        full_code.append(f"{'*'*50}\n# File: {file_pattern}\n{'*'*50}\n")
        full_code.append(code_str)
    return "\n".join(full_code)

def check_fetch_repo(path):
    og_path = path
    if not os.path.exists(path):
        path = path.split("/")[-1].replace(" ", "-")
        repo_url = GITHUB_PREFIX + path
        print(f"{path} not found, cloning with git: {repo_url} into {og_path}")
        subprocess.run(['git', 'clone', repo_url, f"{og_path}"])
    else:
        print(f"{path} exists, using cached version")

if __name__=="__main__":
    path1 = SUBMISSIONS_ROOT_DIR + sys.argv[1]
    path2 = SUBMISSIONS_ROOT_DIR + sys.argv[2]
    check_fetch_repo(path1)
    check_fetch_repo(path2)

    text1 = get_code(path1)
    text2 = get_code(path2)

    email = f"your_email_{random.randrange(0,1000)}@example.com"
    generate_diff(text1, text2, email)
