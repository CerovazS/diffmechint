"""Build an ImageFolder layout for ImageNet val using bbox-derived labels.

Source: 50k flat JPEGs in <SRC_IMG_DIR> and 50k XMLs in <BBOX_XML_DIR> (one per
image). Each XML has at least one <object><name>nXXXXXXXX</name>; we use the
first as the image's canonical class label (the ILSVRC convention).

Output: <DST_DIR>/nXXXXXXXX/ILSVRC2012_val_NNNNNNNN.JPEG  (symlinks, no copy)

Run on the login node once.
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src-img-dir", type=Path, required=True)
    p.add_argument("--bbox-xml-dir", type=Path, required=True)
    p.add_argument("--dst-dir", type=Path, required=True)
    args = p.parse_args()

    xmls = sorted(args.bbox_xml_dir.glob("*.xml"))
    if len(xmls) != 50000:
        print(f"WARN: expected 50000 XMLs, found {len(xmls)}", file=sys.stderr)

    # Parse each XML, extract first <object><name>.
    mapping: dict[str, str] = {}  # filename_stem -> synset
    for xml_path in xmls:
        root = ET.parse(xml_path).getroot()
        obj = root.find("object")
        if obj is None:
            print(f"WARN: no <object> in {xml_path.name}", file=sys.stderr)
            continue
        name = obj.findtext("name")
        if not name:
            print(f"WARN: no <name> in {xml_path.name}", file=sys.stderr)
            continue
        mapping[xml_path.stem] = name

    print(f"parsed {len(mapping)} labels from {len(xmls)} XMLs")

    # Build symlink tree.
    args.dst_dir.mkdir(parents=True, exist_ok=True)
    classes = sorted(set(mapping.values()))
    print(f"unique classes: {len(classes)}")
    for syn in classes:
        (args.dst_dir / syn).mkdir(exist_ok=True)

    n_linked = 0
    n_missing = 0
    for stem, syn in mapping.items():
        src = args.src_img_dir / f"{stem}.JPEG"
        dst = args.dst_dir / syn / f"{stem}.JPEG"
        if not src.exists():
            n_missing += 1
            continue
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        os.symlink(src, dst)
        n_linked += 1

    print(f"linked {n_linked} images; missing source for {n_missing}")
    return 0 if n_missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
