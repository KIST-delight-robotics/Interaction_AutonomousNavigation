#!/usr/bin/env python3
"""
gps_distance_filter.py
  - 마지막으로 보존한 점에서 k 미터 이상 떨어지면 keep
  - 그 사이( < k m ) 값은 모두 삭제

예:
python gps_distance_filter.py --input gps_walk.txt --output gps_walk_2m.txt --k 2
"""
import argparse, csv, math, sys

R = 6_371_000  # Earth radius in metres

def haversine(lat1, lon1, lat2, lon2):
    φ1, φ2 = map(math.radians, (lat1, lat2))
    dφ, dλ = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  "-i", required=True, help="원본 CSV")
    p.add_argument("--output", "-o", required=True, help="저장할 CSV")
    p.add_argument("--k",      type=float, required=True, help="보존 간격(m)")
    return p.parse_args()

def main():
    args = parse_args()
    rows = list(csv.DictReader(open(args.input, newline="")))
    if not rows:
        sys.exit("CSV가 비었습니다.")

    keep = [rows[0]]          # 첫 행은 무조건 보존
    last  = rows[0]

    for row in rows[1:]:
        d = haversine(float(last["lat"]), float(last["lon"]),
                      float(row["lat"]),  float(row["lon"]))
        if d >= args.k:       # k m 이상 떨어지면 보존
            keep.append(row)
            last = row

    # 마지막 원본 행이 포함돼 있지 않으면 추가(선택)
    if keep[-1] is not rows[-1]:
        keep.append(rows[-1])

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(keep)

if __name__ == "__main__":
    main()

