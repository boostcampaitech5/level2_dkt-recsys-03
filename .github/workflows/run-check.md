name: 실행 확인

on:
  pull_request:
    branches: [main]
    
jobs:
  run_seq:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@2
    - name: move to sequential project dir
      run: cd sequential
    - name: setup python
      uses: actions/python@v2
      with:
        python-version: 3.10
    - name: install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: run main.py
      run: python main.py