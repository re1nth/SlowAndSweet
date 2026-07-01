---
description: Show SlowAndSweet usage stats (calls delegated, tokens saved).
---

Run the following bash command and print its output verbatim:

```bash
if command -v slowandsweet >/dev/null 2>&1; then
  slowandsweet stats
else
  echo "slowandsweet binary not found."
  echo "install: pipx install slowandsweet"
fi
```
