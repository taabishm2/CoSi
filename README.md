# cosi
Code Similarity detection for Python files and Jupyter Notebooks

Runs pairwise comparisons between Python code from .py and .ipynb files to determine:

* **Code similarity**: Compares code contents after substituting variable names and running other normalizations. Uses a windowed, fingerprinting approach to come up with vector representations for the submissions, which are then compared using cosine similarity. Based primarily on http://theory.stanford.edu/~aiken/publications/papers/sigmod03.pdf.
* **Output similarity**: Compares similarity between .ipynb output cells (again using the cosine similarity metric)
* **Variable similarity**: Compares similarity between all variable names, including function and class names, using Jaccard similarity calculation.
* **Structure similarity**: Compares the abstract syntax trees (ASTs) for different submissions using Jaccard similarity.

Example output:
```json
{
  "repoA": "https://github.com/cs544-wisc/project-6-repo-A",
  "repoB": "https://github.com/cs544-wisc/project-6-repo-B",
  "code": 0.0258,
  "variables": 0.357,
  "output": 0.9634,
  "struture": 0.7879,
  "visualize": "python3 display.py 'bds-project6' 'professoryuribe'"
}
```

Note: The code within the files must be syntactically valid since CoSi uses AST parsing to extract code structure. 

## Checking code similarity
1. Add a new directory (e.g. `project-10`) within `./files/submissions` containing individual student submission directories
2. Update the `.env` file; set `SUBMISSIONS_ROOT_DIR` as the folder name (`project-10/`) and `FILES_IN_SUBMISSION` as the list of files to be read from each sub-directory of `SUBMISSIONS_ROOT_DIR` e.g (`p10.py` or `nb/client.py,nb/server.py`). Also, set the `GITHUB_PREFIX` as the URL prefix to add to the directory name (used in the resulting CSV) or leave blank if not applicable
3. Run `python3 CoSi.py` 
4. The result is saved as a CSV in `./files/results` identified by the time of generation. A `FAILED` csv is also included if parsing of any files fails

## Viewing code pair diffs
The result csv contains a column to view differences between the code in two submissions. In general, to visualize the text-diff of two submissions, set the `SUBMISSIONS_ROOT_DIR` and `FILES_IN_SUBMISSION` fields in `.env`. Then execute `python3 display.py 'repoA' 'repoB'`, where `repoA` and `repoB` are student directories within the `SUBMISSIONS_ROOT_DIR`. This will open a browser window with a side-by-side comparison of all files from `FILES_IN_SUBMISSION` in the two repos.

Note: If `repoA` and `repoB` are present in `SUBMISSIONS_ROOT_DIR`, the saved version of the files will be displayed. If any of the repo names passed is not saved locally, it will be cloned from git into `SUBMISSIONS_ROOT_DIR` first.
