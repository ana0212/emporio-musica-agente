import shutil
from pathlib import Path

import pytest

from src.repositories.store_repository import StoreRepository


@pytest.fixture
def repository(tmp_path):
    # Set EMporio_DB_PATH to your populated SQLite file, or copy it here.
    source = Path(__file__).resolve().parents[1] / "database" / "emporio_musica.sqlite"
    if not source.exists():
        pytest.skip("Place the populated SQLite database at database/emporio_musica.sqlite")
    target = tmp_path / "test.sqlite"
    shutil.copy2(source, target)
    return StoreRepository(target)


def test_list_categories(repository):
    categories = repository.list_categories()
    assert len(categories) == 9
    assert all("name" in category for category in categories)


def test_search_products_returns_only_active(repository):
    products = repository.search_products(limit=50)
    assert len(products) <= 50
    assert all(product["status"] == "active" for product in products)


def test_search_products_limit_is_bounded(repository):
    products = repository.search_products(limit=10000)
    assert len(products) <= 50


def test_product_details_includes_active_promotions(repository):
    product = repository.get_product_details(96)
    assert product is not None
    assert isinstance(product["active_promotions"], list)


def test_nonexistent_product_returns_none(repository):
    assert repository.get_product_details(-999) is None


def test_order_requires_matching_phone(repository):
    # A deliberately wrong phone must not disclose order information.
    assert repository.get_order_status(1, "00000000000") is None
