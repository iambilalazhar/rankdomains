<<<<<<< HEAD
# rankdomains
A python script to help you find good domains from a csv file of thousands of domains. 
=======
## Domain Filter CLI

Small script that reads `domains.csv` (first column = domain names), uses OpenAI to score meaning/brandability/value, and writes only the best domains to an output CSV.

### Setup

1) Create and activate a virtual environment, then install deps:

```bash
bash setup_venv.sh
source .venv/bin/activate
```

2) Put your domains in `domains.csv`. Only the first column is read; header optional.

3) Set your OpenAI API key in one of these ways:
- Edit `filter_domains.py` and replace `DEFAULT_OPENAI_API_KEY` with your key; or
- Export `OPENAI_API_KEY` in your shell; or
- Pass `--api-key YOUR_KEY` on the command line.

### Run

```bash
python filter_domains.py --input domains.csv --output valuable_domains.csv --min-score 7 --concurrency 8 --temperature 0.0
```

Useful flags:
- `--dry-run` prints a summary without writing output
- `--no-prefilter` disables the quick regex-based filter
- `--model` to change the OpenAI model (default: `gpt-4o-mini`)
- `--batch-size` controls domains per API call (default: 50)
- `--concurrency` runs requests in parallel (default: 1). Try 4–12 for 10k+ domains, adjust to your rate limits.
- `--temperature` sets randomness (default: 0.0). Use 0.1–0.3 if you want slightly more lenient/varied judgments.

### Notes

- The prompt asks for a JSON array with: `domain`, `keep` (boolean), `score` (0-10), `reason`.
- The script prefilters obviously invalid/low-quality domains before calling the API to save tokens; disable with `--no-prefilter` if you want to send everything.
- Output `valuable_domains.csv` includes `domain`, `score`, and `reason` for kept items.
>>>>>>> e7c5259 (Domain evaluator)
