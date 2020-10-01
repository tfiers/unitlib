# How to run tests

1. Install your local version of the `yunit` package, by running,
   in the project root dir:
    ```
   pip install -e .
   ```
2. Install the testing requirements:
   ```
   pip install -r tests/requirements.txt
   ```
3. Finally, run
   ```
   pytest
   ```

These tests are also [run automatically](../.github/workflows/autotest.yml) 
on every push of Python code to GitHub, on a range of different Python versions and OS'es.
