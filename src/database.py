import sqlite3
import json
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "nutrition.db"


def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS children (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            sex INTEGER DEFAULT 0,
            guardian_name TEXT DEFAULT '',
            guardian_contact TEXT DEFAULT '',
            face_encoding BLOB,
            photo BLOB,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            child_id INTEGER REFERENCES children(id) ON DELETE CASCADE,
            age_months INTEGER,
            weight REAL,
            height REAL,
            muac REAL,
            hb REAL,
            bmi REAL,
            wasting_pred INTEGER,
            stunting_pred INTEGER,
            anemia_flag INTEGER,
            risk_level TEXT,
            recommendations TEXT,
            assessed_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()


def add_child(name, sex=0, guardian_name="", guardian_contact="",
              face_encoding=None, photo=None):
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO children (name, sex, guardian_name, guardian_contact,
           face_encoding, photo) VALUES (?, ?, ?, ?, ?, ?)""",
        (name, sex, guardian_name, guardian_contact, face_encoding, photo),
    )
    child_id = cur.lastrowid
    conn.commit()
    conn.close()
    return child_id


def update_child(child_id, **kwargs):
    allowed = {"name", "sex", "guardian_name", "guardian_contact",
               "face_encoding", "photo"}
    fields = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [child_id]
    conn = get_conn()
    conn.execute(
        f"UPDATE children SET {set_clause}, updated_at = datetime('now') WHERE id = ?",
        values,
    )
    conn.commit()
    conn.close()


def get_child(child_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM children WHERE id = ?", (child_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_children():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM children ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_child(child_id):
    conn = get_conn()
    conn.execute("DELETE FROM children WHERE id = ?", (child_id,))
    conn.commit()
    conn.close()


def add_assessment(child_id, age_months, weight, height, muac, hb, bmi,
                   wasting_pred, stunting_pred, anemia_flag, risk, recommendations):
    recs_json = json.dumps(recommendations) if not isinstance(recommendations, str) else recommendations
    conn = get_conn()
    conn.execute(
        """INSERT INTO assessments
           (child_id, age_months, weight, height, muac, hb, bmi,
            wasting_pred, stunting_pred, anemia_flag, risk_level, recommendations)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (child_id, age_months, weight, height, muac, hb, bmi,
         wasting_pred, stunting_pred, anemia_flag, risk, recs_json),
    )
    conn.commit()
    conn.close()


def get_assessments(child_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM assessments WHERE child_id = ? ORDER BY assessed_at DESC",
        (child_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_latest_assessments(risk_filter=None):
    conn = get_conn()
    query = """
        SELECT c.id as child_id, c.name, c.sex, c.photo,
               a.age_months, a.wasting_pred, a.stunting_pred, a.anemia_flag,
               a.risk_level, a.assessed_at, a.recommendations
        FROM children c
        JOIN assessments a ON a.child_id = c.id
        WHERE a.id = (
            SELECT a2.id FROM assessments a2
            WHERE a2.child_id = c.id
            ORDER BY a2.assessed_at DESC LIMIT 1
        )
    """
    if risk_filter:
        placeholders = ", ".join("?" for _ in risk_filter)
        query += f" AND a.risk_level IN ({placeholders})"
        query += " ORDER BY CASE a.risk_level WHEN 'critical' THEN 0 WHEN 'high' THEN 1 ELSE 2 END"
        rows = conn.execute(query, risk_filter).fetchall()
    else:
        rows = conn.execute(query).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_children_with_latest_risk():
    conn = get_conn()
    rows = conn.execute("""
        SELECT c.id, c.name, c.sex, c.guardian_name,
               (SELECT a.risk_level FROM assessments a WHERE a.child_id = c.id
                ORDER BY a.assessed_at DESC LIMIT 1) as risk_level,
               (SELECT COUNT(*) FROM assessments a WHERE a.child_id = c.id) as assessment_count
        FROM children c
        ORDER BY c.name
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats():
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM children").fetchone()[0]
    risk_counts = conn.execute("""
        SELECT a.risk_level, COUNT(*) as cnt
        FROM children c
        JOIN assessments a ON a.child_id = c.id
        WHERE a.id = (
            SELECT a2.id FROM assessments a2
            WHERE a2.child_id = c.id
            ORDER BY a2.assessed_at DESC LIMIT 1
        )
        GROUP BY a.risk_level
    """).fetchall()
    conn.close()
    counts = {row['risk_level']: row['cnt'] for row in risk_counts}
    return {
        "total": total,
        "critical": counts.get("critical", 0),
        "high": counts.get("high", 0),
        "moderate": counts.get("moderate", 0),
        "low": counts.get("low", 0),
    }


def get_all_face_data():
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, face_encoding FROM children WHERE face_encoding IS NOT NULL"
    ).fetchall()
    conn.close()
    return [(row['id'], row['face_encoding']) for row in rows]
