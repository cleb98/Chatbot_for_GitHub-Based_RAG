import yaml
# === CONFIGURATOR ===
class Configurator:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Validazione base
        self.repo_url = self.config.get("repo_url")
        self.local_repo_dir = self.config.get("local_repo_dir", "repo_tmp")
        self.output_md_path = self.config.get("output_md_path", "repo_parsed.md")
        self.include_extensions = set(self.config.get("include_extensions", [".py", ".md"]))

        if not self.repo_url:
            raise ValueError("repo_url è obbligatorio nel file YAML")
        
    #non sicuro se serve, ma lo lascio per compatibilità
    def get(self, key: str, default=None):
        return self.config.get(key, default)