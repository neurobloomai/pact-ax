"""
Entry point for running the PACT-AX proxy as a package:
    python -m proxy.src

Using this instead of `python -m proxy.src.proxy` avoids the
RuntimeWarning on Python 3.9 caused by proxy.src.proxy being added
to sys.modules before execution as __main__.
"""
import asyncio
from proxy.src.proxy import main

asyncio.run(main())
