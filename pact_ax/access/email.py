"""
pact_ax/access/email.py
────────────────────────
Institutional email validation for free tier registration.

"Institutional" means: not a known free consumer email provider.
We don't require .edu — research labs and startups use custom domains.
We do block the big free providers to keep the signal population meaningful.
"""

import re
from typing import Tuple

# Known free consumer email providers — not exhaustive, grows over time
_FREE_PROVIDERS = {
    "gmail.com", "googlemail.com",
    "yahoo.com", "yahoo.co.uk", "yahoo.fr", "yahoo.de", "yahoo.es",
    "hotmail.com", "hotmail.co.uk", "hotmail.fr", "hotmail.de",
    "outlook.com", "outlook.co.uk",
    "live.com", "live.co.uk",
    "msn.com",
    "icloud.com", "me.com", "mac.com",
    "aol.com",
    "protonmail.com", "proton.me",
    "tutanota.com", "tutamail.com",
    "zoho.com",
    "yandex.com", "yandex.ru",
    "mail.com", "email.com",
    "inbox.com",
    "gmx.com", "gmx.net", "gmx.de",
    "web.de",
    "qq.com", "163.com", "126.com",
}

_EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)


def validate_institutional_email(email: str) -> Tuple[bool, str]:
    """
    Return (is_valid, reason).

    Passes when:
    - Email is syntactically valid
    - Domain is not a known free consumer provider

    Returns the organisation domain on success so the caller can record it.
    """
    email = email.strip().lower()

    if not _EMAIL_RE.match(email):
        return False, "Invalid email format"

    domain = email.split("@", 1)[1]

    if domain in _FREE_PROVIDERS:
        return False, (
            f"{domain} is a free consumer email provider. "
            "Please register with your institutional or organisation email."
        )

    return True, domain
