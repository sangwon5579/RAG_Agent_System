from __future__ import annotations

import json
import urllib.request


def main() -> None:
    payload = {
        "query": "형벌 유형 중 재산형에 포함되지 않는 것은?\nA. 금고\nB. 벌금\nC. 과료\nD. 몰수"
    }
    req = urllib.request.Request(
        "http://127.0.0.1:8000/inference",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8")
        print(body)


if __name__ == "__main__":
    main()
