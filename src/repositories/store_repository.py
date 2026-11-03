"""Data access layer for Empório da Música.

All SQL is parameterized. This module does not contain LLM or Streamlit logic.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class StoreRepository:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {self.db_path}")

        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def list_categories(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT category_id, name, description
                FROM categories
                ORDER BY name
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def search_products(
        self,
        query: str = "",
        category_id: int | None = None,
        in_stock_only: bool = False,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search active products by name, description or category.

        `limit` is bounded to avoid unexpectedly large tool responses.
        """
        limit = max(1, min(int(limit), 50))
        filters = ["p.status = 'active'"]
        params: list[Any] = []

        if query.strip():
            pattern = f"%{query.strip()}%"
            filters.append(
                """(
                    p.name LIKE ? COLLATE NOCASE
                    OR p.description LIKE ? COLLATE NOCASE
                    OR p.specs LIKE ? COLLATE NOCASE
                    OR c.name LIKE ? COLLATE NOCASE
                )"""
            )
            params.extend([pattern, pattern, pattern, pattern])

        if category_id is not None:
            filters.append("p.category_id = ?")
            params.append(int(category_id))

        if in_stock_only:
            filters.append("p.stock_quantity > 0")

        sql = f"""
            SELECT
                p.product_id,
                p.name,
                p.description,
                p.price_brl,
                p.stock_quantity,
                p.status,
                p.specs,
                p.category_id,
                c.name AS category_name
            FROM products AS p
            JOIN categories AS c ON c.category_id = p.category_id
            WHERE {' AND '.join(filters)}
            ORDER BY p.name
            LIMIT ?
        """
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        results = []
        for row in rows:
            item = dict(row)
            try:
                item["specs"] = json.loads(item["specs"]) if item["specs"] else {}
            except (TypeError, json.JSONDecodeError):
                item["specs"] = {}
            results.append(item)
        return results

    def get_product_details(self, product_id: int) -> dict[str, Any] | None:
        """Return product details and every currently active promotion.

        Promotions are returned as a list because the schema permits more than
        one active promotion per product. The caller must not arbitrarily pick
        one or stack discounts.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    p.product_id, p.name, p.description, p.price_brl,
                    p.stock_quantity, p.status, p.specs, p.created_at,
                    p.category_id, c.name AS category_name
                FROM products AS p
                JOIN categories AS c ON c.category_id = p.category_id
                WHERE p.product_id = ?
                """,
                (int(product_id),),
            ).fetchone()

            if row is None:
                return None

            product = dict(row)
            promotions = conn.execute(
                """
                SELECT promotion_id, discount_percent, description
                FROM promotions
                WHERE product_id = ? AND is_active = 1
                ORDER BY promotion_id
                """,
                (int(product_id),),
            ).fetchall()

        try:
            product["specs"] = json.loads(product["specs"]) if product["specs"] else {}
        except (TypeError, json.JSONDecodeError):
            product["specs"] = {}

        product["active_promotions"] = [dict(p) for p in promotions]
        return product

    @staticmethod
    def _normalize_phone(phone: str) -> str:
        return "".join(ch for ch in str(phone) if ch.isdigit())

    def get_order_status(
        self,
        order_id: int,
        customer_phone: str,
    ) -> dict[str, Any] | None:
        """Fetch an order only when its customer's phone matches.

        Phone is used as a basic demo verification factor, not production
        authentication. Do not expose customer records based on email alone:
        the source data contains duplicate email addresses.
        """
        supplied_phone = self._normalize_phone(customer_phone)
        if not supplied_phone:
            return None

        with self._connect() as conn:
            order = conn.execute(
                """
                SELECT
                    o.order_id, o.order_date, o.status, o.total_brl,
                    o.payment_method, o.tracking_code, o.estimated_delivery,
                    o.notes, c.name AS customer_name, c.phone AS customer_phone
                FROM orders AS o
                JOIN customers AS c ON c.customer_id = o.customer_id
                WHERE o.order_id = ?
                """,
                (int(order_id),),
            ).fetchone()

            if order is None:
                return None

            stored_phone = self._normalize_phone(order["customer_phone"] or "")
            if not stored_phone or stored_phone != supplied_phone:
                return None

            items = conn.execute(
                """
                SELECT oi.quantity, p.product_id, p.name AS product_name
                FROM order_items AS oi
                JOIN products AS p ON p.product_id = oi.product_id
                WHERE oi.order_id = ?
                ORDER BY p.name
                """,
                (int(order_id),),
            ).fetchall()

        result = dict(order)
        # Avoid returning the customer's phone to the LLM/UI.
        result.pop("customer_phone", None)
        result["items"] = [dict(item) for item in items]
        return result
