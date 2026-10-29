#!/usr/bin/env python3
"""Install the pinned official dependency environment; never download model weights."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python', default='python3.11', help='Python 3.11 executable used to create the isolated venv')
    p.add_argument('--index-url', help='Optional package mirror; versions and hashes still come from the official frozen lock')
    a = p.parse_args()
    lock = json.loads((ROOT/'repos.lock.json').read_text())['repositories']['openpi']
    upstream = ROOT/lock['path']
    if not upstream.exists():
        subprocess.run(['git', 'clone', lock['url'], str(upstream)], check=True)
        subprocess.run(['git', '-C', str(upstream), 'checkout', '--detach', lock['runtime_commit']], check=True)
    actual = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != lock['runtime_commit']: raise ValueError('Existing OpenPI checkout differs; not resetting it')
    version = subprocess.check_output([a.python, '-c', 'import sys; print("%s.%s" % sys.version_info[:2])'], text=True).strip()
    if version != '3.11': raise ValueError('Python 3.11 required')
    venv = ROOT/'.venv-openpi'
    if not venv.exists(): subprocess.run([a.python, '-m', 'venv', str(venv)], check=True)
    python = str(venv/'bin/python')
    actual_python = subprocess.check_output([python, '-c', 'import sys; print("%s.%s" % sys.version_info[:2])'], text=True).strip()
    if actual_python != '3.11': raise ValueError('Existing .venv-openpi is not Python 3.11; not replacing it')
    subprocess.run([python, '-m', 'ensurepip', '--upgrade'], check=True)
    subprocess.run([python, '-m', 'pip', 'install', 'uv==0.13.6'], check=True)
    env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(venv), UV_LINK_MODE='copy', GIT_LFS_SKIP_SMUDGE='1')
    uv = str(venv/'bin/uv')
    if a.index_url:
        requirements = ROOT/'.deps/openpi-requirements.txt'
        subprocess.run([uv, 'export', '--frozen', '--no-dev', '--format', 'requirements-txt',
                        '--output-file', str(requirements)], cwd=upstream, env=env, check=True,
                       stdout=subprocess.DEVNULL)
        subprocess.run([uv, 'pip', 'sync', '--python', python, '--link-mode', 'copy',
                        '--index-url', a.index_url, str(requirements)], cwd=upstream, env=env, check=True)
    else:
        subprocess.run([uv, 'sync', '--frozen', '--no-dev'], cwd=upstream, env=env, check=True)
    # Sync installs copies, so the official transformers patches cannot modify a
    # shared uv cache or another project environment through hardlinks.
    site = Path(subprocess.check_output([python, '-c', 'import site; print(site.getsitepackages()[0])'], text=True).strip())
    shutil.copytree(upstream/'src/openpi/models_pytorch/transformers_replace', site/'transformers', dirs_exist_ok=True)
    subprocess.run([python, str(ROOT/'scripts/openpi_s1b.py'), 'check-deps'], check=True)


if __name__ == '__main__': main()
