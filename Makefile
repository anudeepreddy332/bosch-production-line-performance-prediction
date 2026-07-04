.PHONY: setup test lint dashboard-data docker-up

setup:
	pip install -r requirements.txt

test:
	PYTHONPATH=. pytest

lint:
	ruff check .

# scripts/ops/export_dashboard_data.py shipped in PF4 (docs/implementation/portfolio_master_plan.md).
# The existence check below is a harmless defensive guard, not a sign the script is still pending.
dashboard-data:
	@if [ -f scripts/ops/export_dashboard_data.py ]; then \
		PYTHONPATH=. python scripts/ops/export_dashboard_data.py; \
	else \
		echo "scripts/ops/export_dashboard_data.py not implemented yet -- see PF4 in docs/implementation/portfolio_master_plan.md"; \
	fi

docker-up:
	docker compose up --build
