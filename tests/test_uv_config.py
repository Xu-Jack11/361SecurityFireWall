import unittest
from pathlib import Path


class UvConfigTests(unittest.TestCase):
    def test_uv_uses_zju_mirror_as_default_index(self):
        config = Path("uv.toml").read_text(encoding="utf-8")

        self.assertIn('index-url = "https://mirrors.zju.edu.cn/pypi/web/simple"', config)

    def test_setup_script_uses_system_python_312_and_venv_python(self):
        script = Path("scripts/setup_uv_env.sh").read_text(encoding="utf-8")

        self.assertIn("uv venv --python /usr/bin/python3.12 --system-site-packages .venv", script)
        self.assertIn(".venv/bin/python", script)


if __name__ == "__main__":
    unittest.main()
