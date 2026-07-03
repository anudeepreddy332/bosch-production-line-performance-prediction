.PHONY: setup test lint dashboard-data docker-up

setup:
	pip install -r requirements.txt

test:
	PYTHONPATH=. pytest

lint:
	ruff check .

# scripts/ops/export_dashboard_data.py is PF4 work (docs/implementation/portfolio_master_plan.md)
# -- not implemented yet. This target is wired up now so the Makefile's PF2 interface is stable;
# it will do the real export once that script exists.
dashboard-data:
	@if [ -f scripts/ops/export_dashboard_data.py ]; then \
		PYTHONPATH=. python scripts/ops/export_dashboard_data.py; \
	else \
		echo "scripts/ops/export_dashboard_data.py not implemented yet -- see PF4 in docs/implementation/portfolio_master_plan.md"; \
	fi

docker-up:
	docker compose up --build
