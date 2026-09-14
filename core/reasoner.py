import hashlib
import logging
import os
import re
import threading
from dataclasses import asdict, dataclass
from typing import List, Literal
from config import get_settings
from core.symbol_normalization import canonical_symbol

from pettachainer.pettachainer import PeTTaChainer


logger = logging.getLogger(__name__)

ProvenanceKind = Literal["document", "query_support", "background", "persisted"]


@dataclass(frozen=True)
class AtomProvenance:
    kind: ProvenanceKind
    case_id: str | None = None
    document_index: int | None = None


@dataclass(frozen=True)
class QueryResult:
    proofs: tuple[str, ...]
    proof_provenance: tuple[dict, ...]


class Reasoner:
    """
    Owns all atomspace operations. Nothing else in the system
    calls add_atom or query directly.

    Responsibilities:
    - Load atomspace from disk on startup
    - Add statements coming from the parser
    - Execute queries and return proof traces
    - Persist new atoms to disk
    """

    def __init__(self):
        cfg = get_settings()
        self._atomspace_path = cfg.atomspace_path
        self._query_timeout = cfg.chaining_timeout
        self._lock = threading.Lock()
        self._handler = PeTTaChainer()
        self._background_files: set[str] = set()
        self._provenance_by_name: dict[str, set[AtomProvenance]] = {}
        self._statement_hashes_by_name: dict[str, set[str]] = {}
        self._load_from_disk()

    def _load_from_disk(self):
        if not os.path.exists(self._atomspace_path):
            os.makedirs(os.path.dirname(self._atomspace_path), exist_ok=True)
            return
        logger.info("Loading atomspace from %s", self._atomspace_path)
        self._load_file(self._atomspace_path, "persisted")
        logger.info("Atomspace loaded.")

    def _load_file(self, path: str, provenance_kind: ProvenanceKind = "persisted"):
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                atom = line.strip()
                if atom:
                    try:
                        self._handler.add_atom(atom)
                        self._register_provenance(atom, AtomProvenance(provenance_kind))
                    except Exception as exc:
                        logger.warning("Skipping atom %r: %s", atom, exc)

    def load_background_file(self, path: str):
        normalized = os.path.abspath(path)
        if normalized in self._background_files:
            return
        logger.info("Loading background atomspace from %s", path)
        self._load_file(path, "background")
        self._background_files.add(normalized)
        logger.info("Background atomspace loaded.")

    def add_statements(
        self,
        statements: List[str],
        provenance: AtomProvenance | None = None,
    ) -> List[str]:
        """
        Add parsed MeTTa statements to the atomspace and persist them.
        Returns the list of successfully added atoms.
        """
        added, _rejected = self.add_statements_report(statements, provenance=provenance)
        return added

    def add_statements_report(
        self,
        statements: List[str],
        provenance: AtomProvenance | None = None,
    ) -> tuple[List[str], List[dict]]:
        """Like add_statements, but also returns structured rejections.

        Returns:
        - added: list of atoms successfully added (normalized)
        - rejected: list of {"stmt": <normalized>, "error": <str>}
        """

        added: List[str] = []
        rejected: List[dict] = []
        with self._lock:
            with open(self._atomspace_path, "a", encoding="utf-8") as f:
                for stmt in statements:
                    clean = " ".join(str(stmt).split())
                    if not clean:
                        continue
                    try:
                        self._handler.add_atom(clean)
                        f.write(clean + "\n")
                        added.append(clean)
                        self._register_provenance(
                            clean,
                            provenance or AtomProvenance("persisted"),
                        )
                    except Exception as exc:
                        err = str(exc)
                        logger.warning("Failed to add atom %r: %s", clean, err)
                        rejected.append({"stmt": clean, "error": err})
        return added, rejected

    def query(self, pln_query: str) -> List[str]:
        """
        Run a PLN query and return proof traces.
        Try exact fact lookup first for grounded queries, then fall back to
        PeTTaChainer proof search with the configured timeout.
        """
        return list(self.query_with_provenance(pln_query).proofs)

    def query_with_provenance(
        self,
        pln_query: str,
        current_case_id: str | None = None,
    ) -> QueryResult:
        """Run a query and report exact source metadata for named proof atoms."""
        exact = self._query_exact_fact(pln_query)
        if exact:
            proofs = exact
        else:
            proofs = self._query_chainer(pln_query)
        return QueryResult(
            proofs=tuple(proofs),
            proof_provenance=tuple(
                self._proof_provenance(proof, current_case_id)
                for proof in proofs
            ),
        )

    def _query_chainer(self, pln_query: str) -> List[str]:
        try:
            result = self._handler.query(pln_query, timeout_sec=self._query_timeout)
            return result if result else []
        except Exception as exc:
            logger.warning("Query failed for %r: %s", pln_query, exc)
            return []

    def _register_provenance(self, statement: str, provenance: AtomProvenance) -> None:
        name = self._extract_statement_name(statement)
        if name:
            self._provenance_by_name.setdefault(name, set()).add(provenance)
            statement_hash = hashlib.sha256(
                " ".join(statement.split()).encode("utf-8")
            ).hexdigest()
            self._statement_hashes_by_name.setdefault(name, set()).add(statement_hash)

    @staticmethod
    def _extract_statement_name(statement: str) -> str:
        match = re.match(r"^\(:\s+([^\s()]+)", statement.strip())
        return match.group(1) if match else ""

    def _proof_provenance(self, proof: str, current_case_id: str | None) -> dict:
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_./-]*", str(proof)))
        atom_names = sorted(tokens.intersection(self._provenance_by_name))
        source_rows = []
        for atom_name in atom_names:
            for source in sorted(
                self._provenance_by_name[atom_name],
                key=lambda item: (item.kind, item.case_id or "", item.document_index or -1),
            ):
                source_rows.append({"atom_name": atom_name, **asdict(source)})

        document_case_ids_by_name = {
            atom_name: {
                source.case_id
                for source in self._provenance_by_name[atom_name]
                if source.kind == "document" and source.case_id
            }
            for atom_name in atom_names
        }
        ambiguous_atom_names = sorted(
            name
            for name, case_ids in document_case_ids_by_name.items()
            if len(case_ids) > 1
            or len(self._statement_hashes_by_name.get(name, ())) > 1
        )
        current_case_atom_names = {
            name
            for name, case_ids in document_case_ids_by_name.items()
            if case_ids == {current_case_id} and name not in ambiguous_atom_names
        }
        current_case_sources = [
            row
            for row in source_rows
            if row["kind"] == "document"
            and row.get("case_id") == current_case_id
            and row["atom_name"] in current_case_atom_names
        ]
        foreign_case_ids = sorted(
            {
                str(row["case_id"])
                for row in source_rows
                if row["kind"] == "document"
                and row.get("case_id")
                and row.get("case_id") != current_case_id
            }
        )
        return {
            "atom_names": atom_names,
            "ambiguous_atom_names": ambiguous_atom_names,
            "sources": source_rows,
            "current_case_grounded": bool(current_case_sources),
            "foreign_case_ids": foreign_case_ids,
            "foreign_only": bool(foreign_case_ids and not current_case_sources),
            "query_support_only": bool(source_rows) and all(
                row["kind"] == "query_support" for row in source_rows
            ),
            "unknown": not source_rows,
        }

    def _query_exact_fact(self, pln_query: str) -> List[str]:
        target = self._extract_grounded_query_atom(pln_query)
        if not target:
            return []

        for path in self._fact_sources():
            match = self._find_exact_atom_in_file(path, target)
            if match:
                return [match]
        for path in self._fact_sources():
            match = self._find_canonical_atom_in_file(path, target)
            if match:
                return [match]
        return []

    def _extract_grounded_query_atom(self, pln_query: str) -> str:
        match = re.fullmatch(r"\(:\s+[$?][^\s]+\s+(\(.+\))\s+[$?][^\s]+\)", pln_query.strip())
        if not match:
            return ""
        atom = match.group(1)
        if "$" in atom or "?" in atom:
            return ""
        return " ".join(atom.split())

    def _fact_sources(self) -> List[str]:
        paths = [self._atomspace_path, *sorted(self._background_files)]
        return [path for path in paths if os.path.exists(path)]

    def _find_exact_atom_in_file(self, path: str, target: str) -> str:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                atom = line.strip()
                if not atom:
                    continue
                body = self._extract_statement_body(atom)
                if body == target:
                    return atom
        return ""

    def _find_canonical_atom_in_file(self, path: str, target: str) -> str:
        target_signature = self._parse_simple_atom(target)
        if not target_signature:
            return ""
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                atom = line.strip()
                if not atom:
                    continue
                body = self._extract_statement_body(atom)
                signature = self._parse_simple_atom(body)
                if not signature:
                    continue
                if self._canonically_same_signature(target_signature, signature):
                    return atom
        return ""

    def _extract_statement_body(self, statement: str) -> str:
        match = re.fullmatch(
            r"\(:\s+[^\s]+\s+(\(.+\))\s+\(STV\s+[^\s]+\s+[^\s]+\)\)",
            statement.strip(),
        )
        if not match:
            return ""
        return " ".join(match.group(1).split())

    def _parse_simple_atom(self, atom: str) -> dict | None:
        match = re.fullmatch(r"\(([A-Za-z][A-Za-z0-9_]*)((?:\s+[^()\s]+)*)\)", atom.strip())
        if not match:
            return None
        head = match.group(1)
        args = [part for part in match.group(2).split() if part]
        return {"head": head, "args": args, "arity": len(args)}

    def _canonically_same_signature(self, left: dict, right: dict) -> bool:
        if left["head"] != right["head"] or left["arity"] != right["arity"]:
            return False
        for l_arg, r_arg in zip(left["args"], right["args"]):
            if canonical_symbol(l_arg) != canonical_symbol(r_arg):
                return False
        return True

    def reset(self):
        """Clear the in-memory atomspace and wipe the persistence file."""
        with self._lock:
            self._handler = PeTTaChainer()
            self._background_files = set()
            self._provenance_by_name = {}
            self._statement_hashes_by_name = {}
            if os.path.exists(self._atomspace_path):
                os.remove(self._atomspace_path)
        logger.info("Atomspace reset.")

    @property
    def size(self) -> int:
        """Approximate atom count (line count of persistence file)."""
        if not os.path.exists(self._atomspace_path):
            return 0
        with open(self._atomspace_path) as f:
            return sum(1 for line in f if line.strip())

    @property
    def background_size(self) -> int:
        total = 0
        for path in self._background_files:
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as handle:
                total += sum(1 for line in handle if line.strip())
        return total
