"""One controller per SQLite database, including across Uvicorn processes."""

import os
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path


@contextmanager
def controller_lease(path: Path, namespace: str) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use a stable sibling file; unlinking a locked file would let another
    # process create a new inode and become a second controller.
    with path.with_name(path.name + ".lock").open("a+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Cannot acquire the controller lock for {path}") from exc

        with closing(sqlite3.connect(path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS deployment (namespace TEXT PRIMARY KEY)")
            saved = db.execute("SELECT namespace FROM deployment").fetchone()
            if saved and saved[0] != namespace:
                raise ValueError("This state database belongs to another deployment namespace")
            db.execute("INSERT OR IGNORE INTO deployment VALUES (?)", (namespace,))
        yield
