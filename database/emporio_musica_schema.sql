PRAGMA foreign_keys = ON;

CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT
);

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    phone TEXT,
    email TEXT,
    city TEXT
);

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    price_brl REAL NOT NULL CHECK (price_brl > 0),
    name TEXT NOT NULL,
    category_id INTEGER NOT NULL,
    description TEXT,
    stock_quantity INTEGER NOT NULL CHECK (stock_quantity >= 0),
    status TEXT NOT NULL CHECK (status IN ('active', 'discontinued', 'coming_soon')),
    specs TEXT,
    created_at TEXT,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('delivered', 'shipped', 'confirmed', 'pending', 'cancelled')),
    total_brl REAL NOT NULL CHECK (total_brl >= 0),
    payment_method TEXT NOT NULL CHECK (payment_method IN ('pix', 'credit_3x', 'credit_6x', 'credit_12x', 'boleto', 'debit')),
    tracking_code TEXT,
    estimated_delivery TEXT,
    notes TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE order_items (
    order_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    product_id INTEGER NOT NULL,
    PRIMARY KEY (order_id, product_id),
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE promotions (
    promotion_id INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL,
    discount_percent REAL NOT NULL CHECK (discount_percent >= 0 AND discount_percent <= 100),
    description TEXT,
    is_active INTEGER NOT NULL CHECK (is_active IN (0, 1)),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_products_status_stock ON products(status, stock_quantity);
CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_order_items_product ON order_items(product_id);
CREATE INDEX idx_promotions_product_active ON promotions(product_id, is_active);
