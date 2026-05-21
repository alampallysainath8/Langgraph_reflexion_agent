"""
thumbnail_agent/__init__.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Package init — loads .env so every submodule sees the API keys
without needing to call load_dotenv() again.
"""

from dotenv import load_dotenv

load_dotenv()
