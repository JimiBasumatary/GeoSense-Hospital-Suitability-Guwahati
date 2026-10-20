        candidate = text.strip()

        # Accept JSON wrapped in Markdown code fences.
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            candidate = "\n".join(lines).strip()

        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            return []