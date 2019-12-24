
rm -rf ./htmlcov
pytest --cov-report html --cov=tests --cov-branch tests
