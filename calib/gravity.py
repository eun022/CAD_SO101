"""
중력 보상 페이로드(scaling_factor, kg) 설정
사용법: python calib/gravity.py            현재 값만 조회
        python calib/gravity.py <kg>       값 변경 후 다시 읽어 확인 (0이면 기본값으로 복귀)

- scaling_factor는 배율이 아니라 장치 자체 무게 외에 추가로 보상할 페이로드 질량(kg)
  (Haply Inverse Service 3.5.6 릴리스 노트)
- 장치 전체 설정이므로 다른 프로그램에도 적용됨
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트 (2026-11-18 폴더 정리)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import json
import sys
import urllib.request

URL = "http://localhost:10001/inverse3/0144/config/gravity_compensation"
KG_MAX = 0.3  # 이보다 큰 값은 거부 (스타일러스 무게 범위를 크게 넘음)


def request(method, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(URL, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def main():
    print("현재:", request("GET")["data"])
    if len(sys.argv) == 1:
        return
    kg = float(sys.argv[1])
    if not 0.0 <= kg <= KG_MAX:
        print(f"0 ~ {KG_MAX} kg 범위만 허용합니다.")
        return
    res = request("POST", {"enabled": True, "scaling_factor": kg})
    if not res.get("ok"):
        print("변경 실패:", res)
        return
    print("변경 후:", request("GET")["data"])


if __name__ == "__main__":
    main()
