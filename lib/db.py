import sqlite3
import os

table_frames = "CREATE TABLE frames(id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, filename VARCHAR NOT NULL UNIQUE, path TEXT NOT NULL UNIQUE, extracted BOOL DEFAULT false NOT NULL, face_found BOOL DEFAULT false NOT NULL, converted BOOL DEFAULT false NOT NULL);"

def create(filename='faceswap.db'):
    conn = sqlite3.connect(filename)
    cur = conn.cursor()
    cur.execute(table_frames)
    conn.commit()
    conn.close()

def close_connection(conn):
    conn.close()

def open_connection(filename='faceswap.db'):
    conn = sqlite3.connect(filename)
    return conn

def record_extraction(conn, filename, face_found):
    cur = conn.cursor()
    face_found = str(face_found).lower()
    try:
        cur.execute('INSERT INTO frames (filename, path, extracted, face_found) \
                VALUES (?, ?, "true", ?)', (os.path.basename(filename), filename, face_found))
    except sqlite3.IntegrityError:
        cur.execute('UPDATE frames SET \
                extracted="true", \
                face_found=? \
                WHERE filename = ?;', (face_found, filename))

    conn.commit()

def record_conversion(conn, filename, converted):
    cur = conn.cursor()
    cur.execute('UPDATE frames SET \
            converted=? \
            WHERE filename = ?;', (str(converted).lower(), os.path.basename(filename)))
    conn.commit()

def get_already_converted(conn):
    cur = conn.cursor();
    cur.execute('SELECT filename FROM frames WHERE converted = "true";')
    return [v[0] for v in cur.fetchall()]

def get_already_extracted(conn):
    cur = conn.cursor();
    cur.execute('SELECT filename FROM frames WHERE extracted = "true";')
    return [v[0] for v in cur.fetchall()]

def get_frames_with_faces(conn):
    cur = conn.cursor();
    cur.execute('SELECT filename FROM frames WHERE face_found = "true";')
    return [v[0] for v in cur.fetchall()]

def flush(conn):
    cur = conn.cursor();
    cur.execute("DELETE FROM frames;")
    conn.commit()

def delete(conn):
    cur = conn.cursor();
    cur.execute("DROP TABLE frames;")
    conn.commit()

def verify(conn):
    cur = conn.cursor();
    try:
        cur.execute("SELECT * FROM frames;")
    except sqlite3.OperationalError:
        return False
    return True
    
