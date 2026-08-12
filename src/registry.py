import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProviderInfo:
    name: str
    path: Path
    run_script: Path
    process: subprocess.Popen | None = None


class Registry:
    def __init__(self, providers_dir: str):
        self.providers_dir = Path(providers_dir)

        self.providers: dict[str, ProviderInfo] = {}

    def discover(self):

        self.providers.clear()

        if not self.providers_dir.exists():
            return

        for entry in self.providers_dir.iterdir():

            if not entry.is_dir():
                continue

            run_script = None

            if os.name == "nt":

                candidate = entry / "run.bat"

                if candidate.exists():
                    run_script = candidate

            else:

                candidate = entry / "run.sh"

                if candidate.exists():
                    run_script = candidate

            if not run_script:
                continue

            self.providers[entry.name] = ProviderInfo(
                name=entry.name,
                path=entry,
                run_script=run_script,
            )

    def start_all(self):

        for provider in self.providers.values():

            if provider.process is not None:
                continue

            provider.process = self._start_provider(provider)

    def _start_provider(self, provider: ProviderInfo) -> subprocess.Popen:

        if os.name == "nt":

            return subprocess.Popen(
                [str(provider.run_script)],
                cwd=provider.path,
                shell=True,
            )

        return subprocess.Popen(
            [str(provider.run_script)],
            cwd=provider.path,
        )

    def stop_all(self):

        for provider in self.providers.values():

            self.stop(provider.name)

    def stop(self, name: str):

        provider = self.providers.get(name)

        if provider is None:
            return

        process = provider.process

        if process is None:
            return

        try:

            if os.name == "nt":

                process.terminate()

            else:

                os.kill(process.pid, signal.SIGTERM)

            process.wait(timeout=5)

        except Exception:

            try:
                process.kill()
            except Exception:
                pass

        provider.process = None

    def restart(self, name: str):

        self.stop(name)

        provider = self.providers[name]

        provider.process = self._start_provider(provider)

    def get(self, name: str) -> ProviderInfo | None:

        return self.providers.get(name)

    def list(self):

        return list(self.providers.keys())

    def list_detailed(self):

        result = []

        for provider in self.providers.values():

            result.append(
                {
                    "name": provider.name,
                    "path": str(provider.path),
                    "running": provider.process is not None,
                    "pid": (provider.process.pid if provider.process else None),
                }
            )

        return result
