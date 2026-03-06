# Handy commands cheat-sheet
	•	Create env: mamba env create -f environment.yml
	•	Activate: conda activate bosch
	•	Remove: mamba env remove -n bosch
	•	Lock: conda-lock lock -f environment.yml -p osx-arm64 -p linux-64 -o conda-lock.yml
	•	Recreate from lock: conda-lock install --name bosch --lock-file conda-lock.yml
	•	Export pip freeze (if needed): pip freeze > requirements.freeze.txt
	•	Generate pip-locked file: pip-compile --generate-hashes requirements.in