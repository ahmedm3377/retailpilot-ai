"""Versioned DDL for one isolated run schema.

`apply_migrations` is repeatable: it records applied versions in
`schema_migrations` and skips them on later runs. Each migration runs in its own
transaction.
"""
from __future__ import annotations

import psycopg
from psycopg import sql

SCHEMA_MARKER = "hp-retail-desk run schema"

MIGRATIONS: list[tuple[int, str]] = [
    (1, """
CREATE TABLE fixture_meta (
    singleton   BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    as_of       DATE NOT NULL,
    currency    TEXT NOT NULL,
    extra       JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE products (
    sku          TEXT PRIMARY KEY,
    product_id   TEXT NOT NULL,
    size         TEXT NOT NULL,
    price_cents  BIGINT NOT NULL CHECK (price_cents >= 0),
    stock        BIGINT NOT NULL CHECK (stock >= 0),
    position     INTEGER NOT NULL,
    extra        JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX products_product_id_idx ON products (product_id);

CREATE TABLE orders (
    order_id      TEXT PRIMARY KEY,
    customer_id   TEXT NOT NULL,
    status        TEXT NOT NULL,
    delivered_on  DATE,
    position      INTEGER NOT NULL,
    extra         JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX orders_customer_idx ON orders (customer_id);

-- One row is one purchased unit.
CREATE TABLE order_lines (
    order_id    TEXT NOT NULL REFERENCES orders (order_id),
    line_id     TEXT NOT NULL,
    sku         TEXT NOT NULL,
    paid_cents  BIGINT NOT NULL CHECK (paid_cents >= 0),
    unopened    BOOLEAN NOT NULL,
    final_sale  BOOLEAN NOT NULL,
    returned    BOOLEAN NOT NULL,
    position    INTEGER NOT NULL,
    extra       JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (order_id, line_id)
);

CREATE SEQUENCE proposal_seq;
CREATE TABLE proposals (
    proposal_id      TEXT PRIMARY KEY DEFAULT 'PROP-' || nextval('proposal_seq'),
    request_id       TEXT NOT NULL,
    customer_id      TEXT NOT NULL,
    order_id         TEXT NOT NULL,
    line_id          TEXT NOT NULL,
    remedy           TEXT NOT NULL CHECK (remedy IN ('refund', 'exchange')),
    amount_cents     BIGINT NOT NULL CHECK (amount_cents >= 0),
    paid_cents       BIGINT NOT NULL CHECK (paid_cents >= 0),
    original_sku     TEXT NOT NULL,
    replacement_sku  TEXT,
    status           TEXT NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending', 'completed', 'invalidated')),
    FOREIGN KEY (order_id, line_id) REFERENCES order_lines (order_id, line_id),
    CHECK ((remedy = 'refund') = (replacement_sku IS NULL)),
    CHECK (remedy = 'refund' OR amount_cents = 0)
);
ALTER SEQUENCE proposal_seq OWNED BY proposals.proposal_id;
CREATE INDEX proposals_request_idx ON proposals (request_id);

-- Bound values never change; status only moves away from 'pending' once.
CREATE FUNCTION proposals_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'proposal % cannot be deleted', OLD.proposal_id;
    END IF;
    IF (NEW.proposal_id, NEW.request_id, NEW.customer_id, NEW.order_id, NEW.line_id,
        NEW.remedy, NEW.amount_cents, NEW.paid_cents, NEW.original_sku, NEW.replacement_sku)
       IS DISTINCT FROM
       (OLD.proposal_id, OLD.request_id, OLD.customer_id, OLD.order_id, OLD.line_id,
        OLD.remedy, OLD.amount_cents, OLD.paid_cents, OLD.original_sku, OLD.replacement_sku) THEN
        RAISE EXCEPTION 'proposal % is immutable', OLD.proposal_id;
    END IF;
    IF OLD.status <> 'pending' AND NEW.status IS DISTINCT FROM OLD.status THEN
        RAISE EXCEPTION 'proposal % is already %', OLD.proposal_id, OLD.status;
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER proposals_guard BEFORE UPDATE OR DELETE ON proposals
    FOR EACH ROW EXECUTE FUNCTION proposals_guard();

CREATE SEQUENCE return_seq;
CREATE TABLE returns (
    return_id    TEXT PRIMARY KEY DEFAULT 'RET-' || nextval('return_seq'),
    request_id   TEXT NOT NULL UNIQUE,
    proposal_id  TEXT NOT NULL UNIQUE REFERENCES proposals (proposal_id),
    customer_id  TEXT NOT NULL,
    order_id     TEXT NOT NULL,
    line_id      TEXT NOT NULL,
    sku          TEXT NOT NULL,
    remedy       TEXT NOT NULL CHECK (remedy IN ('refund', 'exchange')),
    UNIQUE (order_id, line_id),
    FOREIGN KEY (order_id, line_id) REFERENCES order_lines (order_id, line_id)
);
ALTER SEQUENCE return_seq OWNED BY returns.return_id;

CREATE SEQUENCE refund_seq;
CREATE TABLE refunds (
    refund_id     TEXT PRIMARY KEY DEFAULT 'REF-' || nextval('refund_seq'),
    return_id     TEXT NOT NULL UNIQUE REFERENCES returns (return_id),
    request_id    TEXT NOT NULL UNIQUE,
    proposal_id   TEXT NOT NULL UNIQUE REFERENCES proposals (proposal_id),
    customer_id   TEXT NOT NULL,
    order_id      TEXT NOT NULL,
    line_id       TEXT NOT NULL,
    amount_cents  BIGINT NOT NULL CHECK (amount_cents >= 0),
    currency      TEXT NOT NULL
);
ALTER SEQUENCE refund_seq OWNED BY refunds.refund_id;

CREATE SEQUENCE exchange_seq;
CREATE TABLE exchanges (
    exchange_id      TEXT PRIMARY KEY DEFAULT 'EXC-' || nextval('exchange_seq'),
    return_id        TEXT NOT NULL UNIQUE REFERENCES returns (return_id),
    request_id       TEXT NOT NULL UNIQUE,
    proposal_id      TEXT NOT NULL UNIQUE REFERENCES proposals (proposal_id),
    customer_id      TEXT NOT NULL,
    order_id         TEXT NOT NULL,
    line_id          TEXT NOT NULL,
    original_sku     TEXT NOT NULL,
    replacement_sku  TEXT NOT NULL REFERENCES products (sku),
    CHECK (original_sku <> replacement_sku)
);
ALTER SEQUENCE exchange_seq OWNED BY exchanges.exchange_id;

-- Returned originals; never added back to sellable products.stock.
CREATE SEQUENCE quarantine_seq;
CREATE TABLE quarantine (
    quarantine_id  TEXT PRIMARY KEY DEFAULT 'Q-' || nextval('quarantine_seq'),
    return_id      TEXT NOT NULL UNIQUE REFERENCES returns (return_id),
    request_id     TEXT NOT NULL,
    proposal_id    TEXT NOT NULL REFERENCES proposals (proposal_id),
    customer_id    TEXT NOT NULL,
    order_id       TEXT NOT NULL,
    line_id        TEXT NOT NULL,
    sku            TEXT NOT NULL,
    quantity       INTEGER NOT NULL CHECK (quantity = 1)
);
ALTER SEQUENCE quarantine_seq OWNED BY quarantine.quarantine_id;

CREATE TABLE idempotency (
    request_id    TEXT PRIMARY KEY,
    customer_id   TEXT NOT NULL,
    proposal_id   TEXT NOT NULL REFERENCES proposals (proposal_id),
    status        TEXT NOT NULL CHECK (status = 'completed'),
    result        JSONB NOT NULL
);

CREATE TABLE audit_events (
    event_id   BIGSERIAL PRIMARY KEY,
    actor      TEXT NOT NULL,
    tool       TEXT NOT NULL,
    arguments  JSONB NOT NULL DEFAULT '{}',
    outcome    TEXT NOT NULL,
    result     JSONB,
    error      TEXT,
    effects    JSONB NOT NULL DEFAULT '[]'
);

CREATE FUNCTION audit_append_only() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'audit_events is append-only';
END $$;
CREATE TRIGGER audit_append_only BEFORE UPDATE OR DELETE ON audit_events
    FOR EACH ROW EXECUTE FUNCTION audit_append_only();
"""),
]

# Truncated together on reset; RESTART IDENTITY resets the owned ID sequences.
RESETTABLE_TABLES = ("audit_events", "idempotency", "quarantine", "exchanges", "refunds",
                     "returns", "proposals", "order_lines", "orders", "products", "fixture_meta")


def apply_migrations(conn: psycopg.Connection, schema: str) -> list[int]:
    """Create `schema` if needed and apply pending migrations; return the versions applied."""
    ident = sql.Identifier(schema)
    with conn.transaction():
        conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(ident))
        conn.execute(sql.SQL("COMMENT ON SCHEMA {} IS {}").format(ident, sql.Literal(SCHEMA_MARKER)))
        conn.execute(sql.SQL("SET search_path TO {}").format(ident))
        conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY)")
    applied = []
    for version, ddl in MIGRATIONS:
        with conn.transaction():
            # Serialize concurrent migrators of the same schema.
            conn.execute("LOCK TABLE schema_migrations IN EXCLUSIVE MODE")
            done = conn.execute("SELECT 1 FROM schema_migrations WHERE version = %s",
                                (version,)).fetchone()
            if done:
                continue
            conn.execute(ddl)
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
            applied.append(version)
    return applied
