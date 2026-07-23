"""
Static checks on docker/Dockerfile.non_root.

The non_root image is intended for deployment into hardened Kubernetes
clusters where `securityContext.runAsNonRoot: true` is enforced. The
kubelet validates non-root status by parsing the image's USER field as
an integer — a string name like "nobody" is rejected with
CreateContainerConfigError because the kubelet cannot resolve
/etc/passwd inside the image at admission time.
"""

import os
import re

import pytest

DOCKERFILE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "docker",
    "Dockerfile.non_root",
)


def _final_user_directive(dockerfile_text: str) -> str:
    """Return the value of the last `USER` directive in the file."""
    matches = re.findall(r"^USER\s+(\S+)\s*$", dockerfile_text, re.MULTILINE)
    assert matches, "Dockerfile.non_root has no USER directive"
    return matches[-1]


@pytest.mark.skipif(
    not os.path.exists(DOCKERFILE_PATH),
    reason="Dockerfile.non_root not present in this checkout",
)
def test_final_user_directive_is_numeric():
    """The runtime USER must be a numeric UID so kubelet's runAsNonRoot
    admission check (strconv.Atoi) succeeds."""
    with open(DOCKERFILE_PATH, "r", encoding="utf-8") as f:
        contents = f.read()

    final_user = _final_user_directive(contents)

    assert final_user.isdigit(), (
        f"Dockerfile.non_root final USER is {final_user!r}; must be a numeric UID "
        "so Kubernetes' runAsNonRoot admission check can verify non-root status. "
        "See https://kubernetes.io/docs/tasks/configure-pod-container/security-context/"
    )

    assert int(final_user) != 0, (
        f"Dockerfile.non_root final USER is {final_user} (root); the non_root image "
        "must run as a non-zero UID."
    )


@pytest.mark.skipif(
    not os.path.exists(DOCKERFILE_PATH),
    reason="Dockerfile.non_root not present in this checkout",
)
def test_prisma_binary_cache_is_group_writable():
    """/app/.cache holds PRISMA_BINARY_CACHE_DIR, which `prisma generate` writes
    to at container startup. Arbitrary-UID-with-GID-0 deployments (the
    OpenShift restricted SCC pattern this image targets) can only write there
    if it gets the same chgrp 0 / chmod g=u / chmod g+w treatment as
    $PRISMA_PATH, not just the trailing g+rX pass, otherwise prisma generate
    fails with "Can't write to .../prisma-python/binaries"."""
    with open(DOCKERFILE_PATH, "r", encoding="utf-8") as f:
        contents = f.read()

    for pattern in (
        r"chgrp\s+-R\s+0\s+[^\n]*\bapp/\.cache\b",
        r"chmod\s+-R\s+g=u\s+[^\n]*\bapp/\.cache\b",
        r"chmod\s+-R\s+g\+w\s+[^\n]*\bapp/\.cache\b",
    ):
        assert re.search(pattern, contents), (
            f"Dockerfile.non_root is missing {pattern!r} for /app/.cache; "
            "an arbitrary-UID/GID-0 runtime user won't be able to write to "
            "PRISMA_BINARY_CACHE_DIR"
        )
