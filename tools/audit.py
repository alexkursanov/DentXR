"""
Шаг 2. Аудит датасета DenPAR: что лежит в данных и насколько разметка согласована.

Что проверяем
  1. Состав: снимки, маски зубов, рамки, точки CEJ и верхушек, линии кости по частям train/val/test.
  2. Согласованность: число масок = числу рамок? = числу номеров FDI в таблице?
     Совпадают ли размеры масок и снимков? Есть ли файлы без пары?
  3. Привязку точек к зубам: насколько далеко каждая точка CEJ / верхушка / конец линии кости
     от ближайшей маски зуба. От этого зависит, сможем ли мы честно сказать «эта точка — этого зуба».

Запуск (из корня проекта):
  python tools/audit.py --data data/Dataset --out work

Результат:
  work/reports/audit.md         сводка
  work/audit_per_image.csv      строка на снимок — для поиска проблемных случаев
  work/figures/audit_examples.jpg  4 случайных снимка со всей разметкой
"""
import argparse, csv, json, re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SPLITS = ["Training", "Validation", "Testing"]
NEAR_PX = 15          # точка ближе этого к маске — считаем, что она «на зубе»


def load_meta(data):
    """Таблица авторов: челюсть, сторона, номера FDI видимых зубов."""
    df = pd.read_excel(data / "Characteristics of radiographs included.xlsx")
    df.columns = ["id", "arch", "site", "fdi"]
    df["fdi_list"] = df["fdi"].astype(str).apply(lambda s: re.findall(r"\d\d", s))
    df["fdi_odd"] = df["fdi"].astype(str).str.contains(r"[^\d,\s]")
    return df.set_index("id")


def point_distances(points, dts, shape):
    """Для каждой точки: расстояние до ближайшей и до второй по близости маски."""
    H, W = shape
    out = []
    for x, y in points:
        xi, yi = min(max(int(round(x)), 0), W - 1), min(max(int(round(y)), 0), H - 1)
        d = sorted(float(dt[yi, xi]) for dt in dts) + [np.inf]
        out.append((d[0], d[1]))
    return out


def draw(img, masks, kp, bone, title):
    """Снимок со всей разметкой: маски, рамки, CEJ (жёлтые), верхушки (красные), кость (оранж.)."""
    colors = [(230, 90, 90), (90, 200, 90), (90, 90, 230), (220, 200, 60), (200, 90, 200), (60, 200, 200)]
    over = img.copy()
    for i, m in enumerate(masks):
        over[m] = colors[i % len(colors)]
    img = cv2.addWeighted(img, 0.6, over, 0.4, 0)
    for x0, y0, x1, y1 in kp["bboxes"]:
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), 2)
    for p in kp["CEJ_Points"]:
        cv2.circle(img, tuple(map(int, p)), 9, (0, 255, 255), -1)
    for p in kp["Apex_Points"]:
        cv2.circle(img, tuple(map(int, p)), 9, (0, 0, 255), -1)
    for line in bone["Bone_Lines"]:
        cv2.polylines(img, [np.int32(line)], False, (0, 165, 255), 4)
    cv2.putText(img, title, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
    return cv2.resize(img, (800, int(800 * img.shape[0] / img.shape[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Dataset")
    ap.add_argument("--out", default="work")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    (out / "reports").mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(exist_ok=True)

    meta = load_meta(data)
    rows, dist = [], {"CEJ": [], "Apex": [], "Bone": []}
    issues = Counter()
    for split in SPLITS:
        d = data / split
        images = sorted((d / "Images").glob("*.jpg"), key=lambda p: int(p.stem))
        for img_path in images:
            sid = img_path.stem
            kp_path = d / "Key Points Annotations" / f"{sid}.json"
            bl_path = d / "Bone Level Annotations" / f"{sid}.json"
            mdir = d / "Masks (Tooth-wise)" / sid
            if not (kp_path.exists() and bl_path.exists() and mdir.exists()):
                issues["нет файла разметки"] += 1
                continue
            kp, bone = json.load(open(kp_path)), json.load(open(bl_path))
            h, w = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).shape
            masks = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0 for p in sorted(mdir.glob("*.png"))]
            if any(m.shape != (h, w) for m in masks):
                issues["размер маски ≠ размеру снимка"] += 1
                continue
            # расстояние от каждого пикселя до маски зуба (0 внутри маски)
            dts = [cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5) for m in masks]
            dc = point_distances(kp["CEJ_Points"], dts, (h, w))
            da = point_distances(kp["Apex_Points"], dts, (h, w))
            ends = [p for line in bone["Bone_Lines"] for p in (line[0], line[-1])]
            db = point_distances(ends, dts, (h, w))
            dist["CEJ"] += dc; dist["Apex"] += da; dist["Bone"] += db

            m = meta.loc[int(sid)] if int(sid) in meta.index else None
            n_fdi = len(m["fdi_list"]) if m is not None else -1
            if m is None:
                issues["снимка нет в таблице"] += 1
            elif m["fdi_odd"]:
                issues[f"странная запись FDI: {m['fdi']}"] += 1
            rows.append(dict(
                id=sid, split=split, w=w, h=h,
                arch=m["arch"] if m is not None else "", site=m["site"] if m is not None else "",
                n_masks=len(masks), n_boxes=len(kp["bboxes"]), n_fdi=n_fdi,
                n_cej=len(kp["CEJ_Points"]), n_apex=len(kp["Apex_Points"]), n_bone_lines=len(bone["Bone_Lines"]),
                cej_far=sum(x[0] > NEAR_PX for x in dc), apex_far=sum(x[0] > NEAR_PX for x in da),
                bone_end_far=sum(x[0] > NEAR_PX for x in db),
                masks_overlap=int(sum(int(x.sum()) for x in masks) > int(np.logical_or.reduce(masks).sum()) + 50)
                if masks else 0))

    df = pd.DataFrame(rows)
    df.to_csv(out / "audit_per_image.csv", index=False, encoding="utf-8-sig")

    # ---------- отчёт
    L = ["# Аудит DenPAR", "", "## 1. Состав", "",
         "| часть | снимков | зубов (масок) | точек CEJ | верхушек | линий кости |", "|---|---|---|---|---|---|"]
    for s in SPLITS + ["всего"]:
        g = df if s == "всего" else df[df.split == s]
        L.append(f"| {s} | {len(g)} | {g.n_masks.sum()} | {g.n_cej.sum()} | {g.n_apex.sum()} | {g.n_bone_lines.sum()} |")
    L += ["", "Зубов на снимке: " + ", ".join(f"{k} — {v}" for k, v in sorted(Counter(df.n_masks).items())),
          "", "Челюсть и область: " + ", ".join(f"{a} {s} — {n}" for (a, s), n in
                                                df.groupby(["arch", "site"]).size().items()),
          "", "Размеры снимков (5 частых): " + ", ".join(f"{w}×{h} — {n}" for (w, h), n in
                                                      Counter(zip(df.w, df.h)).most_common(5)), ""]

    L += ["## 2. Согласованность разметки", "",
          f"- масок ≠ рамок: **{(df.n_masks != df.n_boxes).sum()}** снимков",
          f"- номеров FDI в таблице ≠ масок: **{(df.n_fdi != df.n_masks).sum()}** снимков "
          f"(разница: {dict(sorted(Counter(df.n_fdi - df.n_masks).items()))})",
          f"- маски зубов перекрываются: **{df.masks_overlap.sum()}** снимков",
          f"- точек CEJ на зуб в среднем: **{df.n_cej.sum() / df.n_masks.sum():.2f}** "
          "(у каждого зуба 2 стороны — значит, у части зубов CEJ не отмечен)",
          f"- верхушек на зуб в среднем: **{df.n_apex.sum() / df.n_masks.sum():.2f}** (многокорневые зубы дают >1)"]
    L += [f"- {k}: **{v}**" for k, v in issues.items()]
    L.append("")

    L += ["## 3. Можно ли привязать точки к зубам", "",
          "Расстояние от точки до ближайшей маски зуба, пиксели (0 — точка внутри маски).", "",
          "| точки | всего | медиана | 90% | 99% | максимум | дальше {0} px | неоднозначных* |".format(NEAR_PX),
          "|---|---|---|---|---|---|---|---|"]
    for k, v in dist.items():
        v = np.array(v)
        amb = np.mean((v[:, 1] - v[:, 0]) < 5) if len(v) else 0
        L.append(f"| {k} | {len(v)} | {np.median(v[:, 0]):.1f} | {np.percentile(v[:, 0], 90):.1f} | "
                 f"{np.percentile(v[:, 0], 99):.1f} | {v[:, 0].max():.0f} | {(v[:, 0] > NEAR_PX).mean():.1%} | {amb:.1%} |")
    L += ["", "\\* неоднозначная — вторая по близости маска не дальше первой + 5 px (точка между зубами).",
          "", "Bone — концы линий гребня кости: линия идёт между соседними зубами, "
          "а её концы должны упираться в корни. Далёкие концы — линии у края снимка или без зуба рядом.", ""]
    (out / "reports" / "audit.md").write_text("\n".join(L), encoding="utf-8")

    # ---------- картинка с примерами
    tiles = []
    for _, r in df.sample(4, random_state=a.seed).iterrows():
        d = data / r.split
        img = cv2.imread(str(d / "Images" / f"{r.id}.jpg"))
        masks = [cv2.imread(str(p), 0) > 0 for p in sorted((d / "Masks (Tooth-wise)" / r.id).glob("*.png"))]
        kp = json.load(open(d / "Key Points Annotations" / f"{r.id}.json"))
        bone = json.load(open(d / "Bone Level Annotations" / f"{r.id}.json"))
        tiles.append(draw(img, masks, kp, bone, f"{r.id}: {r.arch} {r.site}"))
    hmax = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    cv2.imwrite(str(out / "figures" / "audit_examples.jpg"), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    print(f"Готово: {out / 'reports' / 'audit.md'}")


if __name__ == "__main__":
    main()
