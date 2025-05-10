all: install run

install:
	pip install -r requirements.txt

run:
	python snake-ai-pytorch/complex_snake_obstacles/agent_complex_obs.py


