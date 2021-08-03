test:
	coverage run -m pytest
	coverage report

html:
	coverage run -m pytest
	coverage html
	open htmlcov/index.html