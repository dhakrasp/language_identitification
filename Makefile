venv:
	virtualenv -p python3 venv

dep:
	pip install -r requirements.txt

activate:
	source venv/bin/activate