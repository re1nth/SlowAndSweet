---
description: Run the SlowAndSweet daemon health check.
---

Run the following bash command and print its output verbatim:

```bash
if command -v slowandsweet >/dev/null 2>&1; then
  slowandsweet doctor
else
  echo "slowandsweet binary not found."
  echo "install: pipx install slowandsweet"
fi
```
