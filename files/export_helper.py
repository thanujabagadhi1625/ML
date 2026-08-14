from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from data_processing import _load_export_dataframe, _normalize_export_submissions
from pathlib import Path
from leetcode_fetcher import fetch_and_save_submissions


BROWSER_EXPORT_SCRIPT = r"""
// Paste this into the browser console on your LeetCode submissions page.
(() => {
  const rows = Array.from(document.querySelectorAll('tr')).filter(r => r.querySelectorAll('td').length >= 6);
  const submissions = rows.map(r => {
    const cells = r.querySelectorAll('td');
    const titleCell = cells[1];
    const title = titleCell ? titleCell.innerText.trim() : '';
    const question_id = title || titleCell?.querySelector('a')?.innerText?.trim() || '';
    const difficulty = cells[2]?.innerText.trim() || 'Medium';
    const status = cells[3]?.innerText.trim() || '';
    const runtimeText = cells[4]?.innerText.trim() || '';
    const runtime_ms = Number((runtimeText.match(/\d+/) || [0])[0]);
    const timestamp = cells[5]?.innerText.trim() || '';
    const language = cells[6]?.innerText.trim() || 'Python3';
    return {
      user_id: 1,
      question_id,
      title,
      difficulty,
      status,
      timestamp,
      runtime_ms,
      language,
    };
  });
  copy(JSON.stringify({ submissions }, null, 2));
  console.log('Copied export JSON to clipboard. Paste it into a file called leetcode_history.json.');
})();
"""


def install_leetcode_export() -> None:
    if shutil.which('leetcode-export'):
        print('leetcode-export CLI is already available.')
        return

    print('Installing leetcode-export in the current Python environment...')
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'leetcode-export'], check=True)
    if not shutil.which('leetcode-export'):
        raise RuntimeError('leetcode-export installation succeeded, but the CLI was not found.')


def run_leetcode_export(output_path: Path | None = None) -> Path:
    output_path = output_path or Path('leetcode_history.json')
    if not shutil.which('leetcode-export'):
        install_leetcode_export()

    print('Running leetcode-export...')
    subprocess.run(['leetcode-export'], check=True)

    if not output_path.exists():
        raise FileNotFoundError(f'Expected export file not found: {output_path}')

    return output_path


def normalize_export(export_path: Path, output_path: Path | None = None) -> Path:
    output_path = output_path or Path('leetcode_history_normalized.json')
    raw_df = _load_export_dataframe(export_path)
    normalized_df = _normalize_export_submissions(raw_df)
    payload = {'submissions': normalized_df.to_dict(orient='records')}
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    print(f'Normalized export saved to: {output_path}')
    return output_path


def validate_export(export_path: Path) -> None:
    raw_df = _load_export_dataframe(export_path)
    normalized_df = _normalize_export_submissions(raw_df)
    print('Raw export columns:', raw_df.columns.tolist())
    print('Normalized export columns:', normalized_df.columns.tolist())
    print('\nSample normalized rows:')
    print(normalized_df.head(5).to_json(orient='records', indent=2, date_format='iso'))
    print('\nUse the normalized file with: python main.py --export path/to/file.json')


def print_browser_snippet() -> None:
    print('Copy this JavaScript into your browser console on the LeetCode submissions page:')
    print(BROWSER_EXPORT_SCRIPT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Helper tools for exporting and normalizing LeetCode data.')
    parser.add_argument('--download', action='store_true', help='Install and run leetcode-export to download submissions.')
    parser.add_argument('--fetch', action='store_true', help='Fetch submissions from LeetCode using LEETCODE_SESSION cookie (one-time).')
    parser.add_argument('--normalize', type=Path, help='Normalize an existing export file to the repo schema.')
    parser.add_argument('--output', type=Path, help='Output path for normalized JSON.')
    parser.add_argument('--validate', type=Path, help='Validate an export file and show sample normalized rows.')
    parser.add_argument('--browser-snippet', action='store_true', help='Print a browser console script to extract submissions from LeetCode.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.browser_snippet:
        print_browser_snippet()
        return

    if args.download:
        path = run_leetcode_export()
        print(f'Export completed: {path}')

    if args.fetch:
        out = fetch_and_save_submissions()
        print(f'Fetched submissions saved to: {out}')

    if args.normalize:
        normalize_export(args.normalize, args.output)

    if args.validate:
        validate_export(args.validate)

    if not (args.download or args.normalize or args.validate or args.browser_snippet):
        print('No action specified. Use --help for usage information.')


if __name__ == '__main__':
    main()
