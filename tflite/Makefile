default:
	@echo "Default target not supported - use one of: lint, env" && exit 1

lint:
	pylint -rn -f parseable prepare_data.py

env:
	virtualenv env
	. env/bin/activate && pip install -r requirements.txt
