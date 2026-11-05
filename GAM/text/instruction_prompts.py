"""Per-episode text prompts recovered from a templated instruction (RoboTwin 2.0).

RoboTwin writes every instruction from a task template ("Place {A} on the left of {B}.") by filling each
object placeholder with "the <description>", the description drawn from that object's description file
(description/objects_description/<id>/<variant>.json, lists "seen" / "unseen"), and each arm placeholder
({a}) with "the left arm" / "the right arm". Matching the instruction against the task's templates
recovers the description behind each placeholder; the description file it belongs to names the object
(057_toycar -> "toycar", the name the RoboTwin detector was fine-tuned with). Only the instruction text
and the benchmark's public description files are read.

Config (data section):
    slots: [object, target]
    prompts_from_instruction:
      templates: ${ROBOTWIN_ROOT}/description/task_instruction/place_a2b_left.json
      objects: ${ROBOTWIN_ROOT}/description/objects_description
      placeholders: {object: A, target: B}     # slot -> template placeholder; other slots keep their own name
"""
import glob
import json
import os
import re


def object_name(folder: str) -> str:
    """Description-folder name -> detector prompt: 057_toycar -> toycar, 005_french-fries -> french-fries."""
    nm = re.sub(r"^\d+_", "", folder)
    nm = re.sub(r"_?\d+$", "", nm)
    return nm.replace("_", " ").strip().lower()


ARM = r"the (?:left|right) arm"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).rstrip(" .!")


class InstructionPrompts:
    def __init__(self, spec: dict):
        tj = json.load(open(spec["templates"]))
        self.templates = list(dict.fromkeys(list(tj.get("seen", [])) + list(tj.get("unseen", []))))
        assert self.templates, f"no templates in {spec['templates']}"
        self.placeholders = dict(spec["placeholders"])
        self.desc = {}                                   # description -> {object names}
        for f in glob.glob(os.path.join(spec["objects"], "*", "*.json")):
            name = object_name(os.path.basename(os.path.dirname(f)))
            try:
                d = json.load(open(f))
            except Exception:
                continue
            for s in list(d.get("seen", [])) + list(d.get("unseen", [])) + [d.get("raw_description", "")]:
                if s:
                    self.desc.setdefault(_norm(s), set()).add(name)
        assert self.desc, f"no object descriptions under {spec['objects']}"
        self._by_len = sorted(self.desc, key=len, reverse=True)
        self.regex = [self._compile(t) for t in self.templates]

    @staticmethod
    def _compile(t):
        parts, seen = [], set()
        t = re.sub(r"\s+", " ", t.strip()).rstrip(" .!")       # placeholders keep their case: {A} is an object, {a} an arm
        for tok in re.split(r"(\{[A-Za-z]\})", t):
            m = re.fullmatch(r"\{([A-Za-z])\}", tok)
            if m:
                k = m.group(1)
                grp = ARM if k.islower() else ".+?"       # arm placeholders are always "the left / right arm"
                parts.append(f"(?P={k})" if k in seen else f"(?P<{k}>{grp})"); seen.add(k)
            else:
                parts.append(re.escape(tok.lower()))
        return re.compile("^" + "".join(parts) + "$")

    def _lookup(self, value):
        """Captured span -> (object name, exact); falls back to the longest known description inside the span."""
        v = _norm(value)
        v = v[4:] if v.startswith("the ") else v
        names, exact = self.desc.get(v), True
        if not names:
            hits = [d for d in self._by_len if f" {d} " in f" {v} "]
            names, exact = (self.desc[hits[0]] if hits else None), False
        return (sorted(names)[0] if names else None), exact   # a description shared by several objects: alphabetical first

    def parse(self, instruction: str) -> dict:
        """Instruction -> {placeholder: object name}; the template whose object fill-ins all resolve wins."""
        s, best, best_key = _norm(instruction), {}, None
        for rx in self.regex:
            m = rx.match(s)
            if not m:
                continue
            res = {k: self._lookup(v) for k, v in m.groupdict().items() if k.isupper()}
            key = (sum(e and n is not None for n, e in res.values()), sum(n is not None for n, _ in res.values()), len(res))
            if best_key is None or key > best_key:      # most exact descriptions, then most resolved, then most placeholders
                best, best_key = {k: n for k, (n, _) in res.items()}, key
        return best

    def __call__(self, instruction: str, slots, base_prompts):
        """Prompts for this episode: the parsed object names for the mapped slots, base_prompts elsewhere."""
        got = self.parse(instruction) if instruction else {}
        out = list(base_prompts)
        for i, s in enumerate(slots):
            ph = self.placeholders.get(s)
            if ph is not None:
                out[i] = got.get(ph)                    # None -> unresolved: the slot stays empty
        return out


def build_instruction_prompts(cfg):
    spec = (cfg.get("data") or {}).get("prompts_from_instruction")
    return InstructionPrompts(spec) if spec else None
