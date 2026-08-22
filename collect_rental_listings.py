# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 02:54:13 2026

@author: berat
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:02:10 2026

@author: berat
"""

import argparse
import asyncio
import random
import re
import hashlib
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple
from urllib.parse import urlparse, urlunparse

import pandas as pd
from PIL import Image
from patchright.async_api import async_playwright, Page   # patchright = drop-in, stealth built-in
from curl_cffi import requests as curl_requests

async def stealth_async(page):
    pass  # no-op — patchright kendi patch'lerini zaten uygular


# -------------------------
# Constants / regex
# -------------------------
BASE = "https://www.hepsiemlak.com"
DEFAULT_SEARCH_URL = "https://www.hepsiemlak.com/ankara-kiralik"

IMG_EXT_RE = re.compile(r"\.(jpg|jpeg|png|webp)(\?|$)", re.IGNORECASE)
MNRESIZE_RE = re.compile(r"/mnresize/\d+/\d+/", re.IGNORECASE)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.hepsiemlak.com/",
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
}


# -------------------------
# Utils
# -------------------------
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", ""))


def parse_money_try(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    x = s.replace(".", "").replace("₺", "").replace("TL", "").strip()
    x = x.replace(",", ".")
    try:
        return float(x)
    except Exception:
        return None


def parse_int_from_text(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = re.search(r"\d+", s.replace(".", ""))
    return int(m.group()) if m else None


def parse_m2_pair(s: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """
    Örnek HTML inner_text:
    '70 m2 / 60 m2'
    İlk sayı brüt, ikinci sayı net.
    """
    if not s:
        return None, None

    cleaned = s.lower().replace(",", ".")
    nums = re.findall(r"(\d+(?:\.\d+)?)", cleaned)

    if not nums:
        return None, None

    gross = float(nums[0])
    net = float(nums[1]) if len(nums) > 1 else None

    # Net m2'nin saçma küçük parse edilmesini engelle
    if net is not None and net < 10:
        net = None

    # Brüt de saçmaysa None yap
    if gross < 10:
        gross = None

    return gross, net


def normalize_rooms(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return re.sub(r"\s+", "", s)


def furnished_to_bool(s: Optional[str]) -> Optional[bool]:
    if not s:
        return None
    t = s.lower()
    if "değil" in t or "degil" in t:
        return False
    if "eşyalı" in t or "esyali" in t:
        return True
    return None


def scrub_pii(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    t = text
    t = re.sub(r"tel:\+?\d+", " ", t)
    t = re.sub(r"\b0?\d{3}\s?\d{3}\s?\d{2}\s?\d{2}\b", " ", t)
    t = re.sub(r"\b\(?0\d{3}\)?\s?\d{3}\s?\d{2}\s?\d{2}\b", " ", t)
    t = re.sub(r"\b\+90\s?\d{3}\s?\d{3}\s?\d{2}\s?\d{2}\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def canonicalize_hepsiemlak_image_url(url: str) -> str:
    """
    Thumbnail / resized URL'leri mümkünse daha büyük/orijinal path'e çevir.
    Örn:
    https://.../mnresize/132/74/ds01/...jpg -> https://.../ds01/...jpg
    """
    if not url:
        return url

    url = url.strip()
    url = url.replace("&amp;", "&")

    if MNRESIZE_RE.search(url):
        url = MNRESIZE_RE.sub("/", url)

    return url


def is_listing_photo(url: str) -> bool:
    if not url:
        return False
    if "hecdn" not in url:
        return False
    return bool(IMG_EXT_RE.search(url))


def make_listing_id(listing_no: Optional[str], url: str) -> str:
    if listing_no:
        return f"hepsiemlak:{listing_no}"
    return f"hepsiemlak:{sha1(normalize_url(url))}"


def make_image_id(listing_id: str, order_index: int) -> str:
    return f"{listing_id}:{order_index}"


def rand_delay(min_s: float, max_s: float):
    time.sleep(random.uniform(min_s, max_s))


async def human_pause(min_s: float, max_s: float):
    await asyncio.sleep(random.uniform(min_s, max_s))


def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


# -------------------------
# Image download/resize
# -------------------------
def download_and_resize(
    url: str,
    out_path: Path,
    resize_max_width: int,
    jpeg_quality: int,
    min_img_width: int,
    retry: int,
    delay_min: float,
    delay_max: float,
) -> bool:
    ensure_parent(out_path)
    url = canonicalize_hepsiemlak_image_url(url)

    for _ in range(retry + 1):
        try:
            r = curl_requests.get(
                url,
                headers=HEADERS,
                impersonate="chrome110",
                timeout=25,
            )
            r.raise_for_status()
            out_path.write_bytes(r.content)

            with Image.open(out_path) as im:
                im = im.convert("RGB")
                w, h = im.size

                if w < min_img_width:
                    out_path.unlink(missing_ok=True)
                    return False

                if w > resize_max_width:
                    new_h = int(h * (resize_max_width / w))
                    im = im.resize((resize_max_width, new_h))

                im.save(out_path, format="JPEG", quality=jpeg_quality, optimize=True)

            return True
        except Exception:
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass
            rand_delay(delay_min, delay_max)

    return False


def image_info(path: Path) -> Dict[str, Any]:
    with Image.open(path) as im:
        w, h = im.size
    return {
        "width": w,
        "height": h,
        "file_size": path.stat().st_size,
        "format": "JPEG",
        "aspect_ratio": round(w / h, 4) if h else None,
    }


# -------------------------
# State / parquet
# -------------------------
def load_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def append_lines(path: Path, lines: List[str]):
    if not lines:
        return
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


def load_set(path: Path) -> Set[str]:
    return set(load_lines(path))


def write_lines(path: Path, lines: List[str]):
    ensure_parent(path)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def append_parquet(path: Path, rows: List[Dict[str, Any]], unique_key: Optional[str] = None):
    if not rows:
        return

    df_new = pd.DataFrame(rows)

    if path.exists():
        df_old = pd.read_parquet(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
        if unique_key and unique_key in df.columns:
            df = df.drop_duplicates(subset=[unique_key], keep="last")
    else:
        df = df_new

    ensure_parent(path)
    df.to_parquet(path, index=False)


# -------------------------
# Browser helpers
# -------------------------
async def accept_cookies_if_present(page: Page):
    selectors = [
        "button#onetrust-accept-btn-handler",
        "button:has-text('Kabul Et')",
        "button:has-text('Tümünü Kabul Et')",
        "button:has-text('Accept')",
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if await loc.count() > 0:
                await loc.click(timeout=1500)
                await human_pause(0.4, 0.8)
                return
        except Exception:
            pass


async def warmup_session(page: Page, search_url: str):
    print("[browser] warming up...")
    await page.goto(BASE, wait_until="domcontentloaded", timeout=60000)
    await human_pause(0.8, 1.4)
    await accept_cookies_if_present(page)

    try:
        await page.mouse.move(300, 200, steps=12)
        await page.mouse.wheel(0, 500)
    except Exception:
        pass

    await human_pause(0.4, 0.8)

    await page.goto(search_url, wait_until="domcontentloaded", timeout=60000)
    await human_pause(0.8, 1.4)
    await accept_cookies_if_present(page)
    print("[browser] warmup completed")


# -------------------------
# Page extractors
# -------------------------
async def extract_spec_table(page: Page) -> Dict[str, str]:
    rows = await page.query_selector_all("table.adv-info-list.property-spec-table tr.spec-item")
    specs = {}
    for row in rows:
        th = await row.query_selector("th")
        td = await row.query_selector("td")
        if not th or not td:
            continue
        key = (await th.inner_text()).strip()
        val = (await td.inner_text()).strip()
        if key and val:
            specs[key] = val
    return specs


async def extract_location(page: Page) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    try:
        parts = await page.locator("address.detail-info-location > div").all_inner_texts()
        parts = [p.strip() for p in parts if p.strip()]
        city = parts[0] if len(parts) > 0 else None
        district = parts[1] if len(parts) > 1 else None
        neighborhood = parts[2] if len(parts) > 2 else None
        return city, district, neighborhood
    except Exception:
        return None, None, None


async def extract_price_try(page: Page) -> Optional[float]:
    selectors = [
        "p.fz24-text.price",
        ".price",
    ]
    for sel in selectors:
        try:
            txt = await page.locator(sel).first.inner_text(timeout=2500)
            val = parse_money_try(txt)
            if val is not None:
                return val
        except Exception:
            pass
    return None


async def collect_img_srcs(page: Page) -> List[str]:
    srcs = await page.eval_on_selector_all(
        "img",
        """
        els => els
            .map(e => e.getAttribute('data-src') || e.getAttribute('src') || e.currentSrc)
            .filter(Boolean)
        """
    )

    cleaned = []
    for s in srcs:
        if not s:
            continue
        if s.startswith("data:image/"):
            continue
        s = canonicalize_hepsiemlak_image_url(s)
        if is_listing_photo(s):
            cleaned.append(s)

    # dedup preserve order
    return list(dict.fromkeys(cleaned))


async def extract_image_urls(page: Page, max_images: int, max_next_clicks: int) -> List[str]:
    seen: List[str] = []

    def add_new(items: List[str]):
        existing = set(seen)
        for item in items:
            if item not in existing:
                seen.append(item)
                existing.add(item)

    await page.wait_for_timeout(1500)

    # İlk ekran + thumbnail DOM'dan gelenlerin hepsini topla
    add_new(await collect_img_srcs(page))

    # Büyük image wrapper'dan da ekstra al
    wrapper_imgs = await page.eval_on_selector_all(
        "div.img-wrapper img",
        """
        els => els
            .map(e => e.getAttribute('data-src') || e.getAttribute('src') || e.currentSrc)
            .filter(Boolean)
        """
    )
    add_new([canonicalize_hepsiemlak_image_url(u) for u in wrapper_imgs if is_listing_photo(canonicalize_hepsiemlak_image_url(u))])

    next_btn = page.locator("div.bottom-swiper-button-next").first
    no_progress_rounds = 0

    if await next_btn.count() > 0:
        for _ in range(max_next_clicks):
            if len(seen) >= max_images:
                break

            before = len(seen)

            try:
                await next_btn.click(timeout=3000)
                await page.wait_for_timeout(1000)

                # Lazy load tetikle
                try:
                    await page.mouse.wheel(0, 200)
                except Exception:
                    pass

                await page.wait_for_timeout(500)

                add_new(await collect_img_srcs(page))

                wrapper_imgs = await page.eval_on_selector_all(
                    "div.img-wrapper img",
                    """
                    els => els
                        .map(e => e.getAttribute('data-src') || e.getAttribute('src') || e.currentSrc)
                        .filter(Boolean)
                    """
                )
                add_new([canonicalize_hepsiemlak_image_url(u) for u in wrapper_imgs if is_listing_photo(canonicalize_hepsiemlak_image_url(u))])

                after = len(seen)
                if after == before:
                    no_progress_rounds += 1
                else:
                    no_progress_rounds = 0

                if no_progress_rounds >= 3:
                    break

            except Exception:
                break

    return seen[:max_images]


async def extract_description(page: Page) -> Optional[str]:
    try:
        raw = await page.locator("div.ql-editor.description-content").first.inner_text(timeout=2500)
        return scrub_pii(raw)
    except Exception:
        return None


# -------------------------
# Harvest listing URLs
# -------------------------
async def harvest_listing_urls(page: Page, search_url: str, max_pages: int) -> List[str]:
    await page.goto(search_url, wait_until="domcontentloaded", timeout=60000)
    content = await page.content()
    print("HTML length:", len(content))
    await asyncio.sleep(2)

    urls: List[str] = []

    for page_idx in range(max_pages):
        print(f"[discover] search page {page_idx + 1}/{max_pages}")
        await page.wait_for_timeout(1500)

        hrefs = await page.eval_on_selector_all(
            "a",
            "els => els.map(e => e.getAttribute('href')).filter(Boolean)"
        )

        matched = []

        for h in hrefs:
            if not h:
                continue

            if h.startswith("/"):
                full_url = BASE + h
            elif h.startswith("http"):
                full_url = h
            else:
                continue

            full_url = normalize_url(full_url)

            if "/daire/" in full_url and "kiralik" in full_url:
                matched.append(full_url)

        matched = list(dict.fromkeys(matched))
        print("eşleşen ilan sayısı:", len(matched))

        urls.extend(matched)
        urls = list(dict.fromkeys(urls))

        next_link = page.locator("a.he-pagination__navigate-text--next").first
        next_count = await next_link.count()

        if next_count == 0:
            print("[discover] next page button not found, stopping.")
            break

        try:
            next_href = await next_link.get_attribute("href")
            print(f"[discover] next href: {next_href}")
            await next_link.click(timeout=5000)
            await page.wait_for_load_state("domcontentloaded", timeout=60000)
            await page.wait_for_timeout(1500)
        except Exception as e:
            print(f"[discover] next click failed: {e}")
            break

    return urls


# -------------------------
# Validation
# -------------------------
def validate_listing(listing_row: Dict[str, Any], image_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    price = listing_row.get("price_try")
    city = listing_row.get("city")
    district = listing_row.get("district")
    rooms = listing_row.get("rooms")

    m2_net = listing_row.get("m2_net")
    m2_gross = listing_row.get("m2_gross")
    m2_value = m2_net if (m2_net is not None and m2_net >= 10) else m2_gross

    desc = listing_row.get("description") or ""

    valid_images = [r for r in image_rows if r.get("is_valid")]
    valid_image_count = len(valid_images)

    has_price = price is not None and price > 0
    has_location = bool(city and district)
    has_rooms = bool(rooms)
    has_m2 = m2_value is not None and m2_value > 0
    has_description = len(desc.strip()) >= 20
    has_min_images = valid_image_count >= 3
    all_images_valid = len(image_rows) > 0 and valid_image_count == len(image_rows)

    is_outlier_price = bool(price is not None and (price < 1000 or price > 500000))
    is_outlier_m2 = bool(m2_value is not None and (m2_value < 10 or m2_value > 1000))

    price_per_m2 = None
    is_outlier_price_per_m2 = False
    if price and m2_value and m2_value > 0:
        price_per_m2 = price / m2_value
        is_outlier_price_per_m2 = price_per_m2 < 100 or price_per_m2 > 100000

    is_train_ready_ml = (
        has_price and
        has_location and
        has_rooms and
        has_m2 and
        not is_outlier_price and
        not is_outlier_m2 and
        not is_outlier_price_per_m2
    )

    is_train_ready_dl = (
        has_price and
        has_location and
        has_min_images
    )

    is_train_ready_multimodal = is_train_ready_ml and is_train_ready_dl and has_description

    reject_reasons = []
    if not has_price:
        reject_reasons.append("missing_price")
    if not has_location:
        reject_reasons.append("missing_location")
    if not has_rooms:
        reject_reasons.append("missing_rooms")
    if not has_m2:
        reject_reasons.append("missing_m2")
    if not has_description:
        reject_reasons.append("description_too_short")
    if not has_min_images:
        reject_reasons.append("not_enough_valid_images")
    if is_outlier_price:
        reject_reasons.append("outlier_price")
    if is_outlier_m2:
        reject_reasons.append("outlier_m2")
    if is_outlier_price_per_m2:
        reject_reasons.append("outlier_price_per_m2")

    return {
        "listing_id": listing_row["listing_id"],
        "has_price": has_price,
        "has_location": has_location,
        "has_rooms": has_rooms,
        "has_m2": has_m2,
        "has_description": has_description,
        "has_min_images": has_min_images,
        "all_images_valid": all_images_valid,
        "valid_image_count": valid_image_count,
        "price_per_m2": price_per_m2,
        "is_outlier_price": is_outlier_price,
        "is_outlier_m2": is_outlier_m2,
        "is_outlier_price_per_m2": is_outlier_price_per_m2,
        "is_train_ready_ml": is_train_ready_ml,
        "is_train_ready_dl": is_train_ready_dl,
        "is_train_ready_multimodal": is_train_ready_multimodal,
        "reject_reason": "|".join(reject_reasons) if reject_reasons else None,
        "validated_at": now_iso(),
    }


# -------------------------
# Parse one listing
# -------------------------
async def parse_listing(
    page: Page,
    url: str,
    max_images: int,
    max_next_clicks: int,
    out_dir: Path,
    resize_max_width: int,
    jpeg_quality: int,
    min_img_width: int,
    retry: int,
    delay_min: float,
    delay_max: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    await page.goto(url, wait_until="domcontentloaded", timeout=60000)
    await human_pause(0.6, 1.1)

    try:
        await page.mouse.wheel(0, 500)
    except Exception:
        pass

    await human_pause(0.4, 0.8)

    title = None
    try:
        title = (await page.locator("h1").first.inner_text(timeout=3000)).strip()
    except Exception:
        pass

    price_try = await extract_price_try(page)
    city, district, neighborhood = await extract_location(page)
    specs = await extract_spec_table(page)
    desc = await extract_description(page)

    listing_no = specs.get("İlan no") or specs.get("İlan No")
    listing_id = make_listing_id(listing_no, url)

    rooms = normalize_rooms(specs.get("Oda Sayısı"))
    bathrooms = parse_int_from_text(specs.get("Banyo Sayısı"))
    m2_gross, m2_net = parse_m2_pair(specs.get("Brüt / Net M2"))
    total_floors = parse_int_from_text(specs.get("Kat Sayısı"))
    floor = specs.get("Bulunduğu Kat")
    building_age = parse_int_from_text(specs.get("Bina Yaşı"))
    heating_type = specs.get("Isınma Tipi") or specs.get("Isıtma")
    fuel_type = specs.get("Yakıt Tipi")
    is_furnished = furnished_to_bool(specs.get("Eşya Durumu") or specs.get("Eşyalı"))
    dues_try = parse_money_try(specs.get("Aidat"))
    home_type = specs.get("Konut Tipi")
    home_shape = specs.get("Konut Şekli")
    updated_at = specs.get("Son Güncelleme")

    image_urls = await extract_image_urls(
        page,
        max_images=max_images,
        max_next_clicks=max_next_clicks
    )

    listing_row = {
        "listing_id": listing_id,
        "url": normalize_url(url),
        "listing_no": listing_no,
        "price_try": price_try,
        "city": city,
        "district": district,
        "neighborhood": neighborhood,
        "rooms": rooms,
        "bathrooms": bathrooms,
        "m2_gross": m2_gross,
        "m2_net": m2_net,
        "building_age": building_age,
        "floor": floor,
        "total_floors": total_floors,
        "heating_type": heating_type,
        "fuel_type": fuel_type,
        "is_furnished": is_furnished,
        "dues_try": dues_try,
        "home_type": home_type,
        "home_shape": home_shape,
        "updated_at": updated_at,
        "title": title,
        "description": desc,
        "image_count": len(image_urls),
        "raw_specs_json": json.dumps(specs, ensure_ascii=False),
        "scraped_at": now_iso(),
    }

    images_rows: List[Dict[str, Any]] = []
    safe_listing_folder = listing_id.replace(":", "_")

    for idx, img_url in enumerate(image_urls):
        img_url = canonicalize_hepsiemlak_image_url(img_url)
        img_path = out_dir / "images" / safe_listing_folder / f"{idx:03d}.jpg"

        ok = download_and_resize(
            img_url,
            img_path,
            resize_max_width=resize_max_width,
            jpeg_quality=jpeg_quality,
            min_img_width=min_img_width,
            retry=retry,
            delay_min=delay_min,
            delay_max=delay_max,
        )

        info = {
            "width": None,
            "height": None,
            "file_size": None,
            "format": None,
            "aspect_ratio": None,
        }

        if ok:
            try:
                info = image_info(img_path)
            except Exception:
                ok = False

        images_rows.append({
            "image_id": make_image_id(listing_id, idx),
            "listing_id": listing_id,
            "order_index": idx,
            "source_url": img_url,
            "local_path": str(img_path),
            "is_valid": bool(ok),
            **info,
            "scraped_at": now_iso(),
        })

    validation_row = validate_listing(listing_row, images_rows)

    return listing_row, images_rows, validation_row


# -------------------------
# Main run
# -------------------------
async def run(args):
    out_dir = Path(args.out_dir)
    state_dir = Path(args.state_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    visited_path = state_dir / "visited.txt"
    queue_path = state_dir / "queue.txt"

    runlog_path = out_dir / "run_log.parquet"
    listings_path = out_dir / "listings.parquet"
    images_path = out_dir / "images.parquet"
    validation_path = out_dir / "validation_report.parquet"
    train_ready_ml_path = out_dir / "train_ready_ml.parquet"
    train_ready_multimodal_path = out_dir / "train_ready_multimodal.parquet"

    visited = load_set(visited_path)
    queue = load_lines(queue_path)

    run_id = sha1(now_iso() + str(random.random()))[:12]

    def log(event: Dict[str, Any]):
        event = {"run_id": run_id, **event}
        append_parquet(runlog_path, [event])

    log({
        "ts": now_iso(),
        "event": "run_start",
        "search_url": args.search_url,
        "limit": args.limit,
        "headful": args.headful,
    })

    async with async_playwright() as p:
        # YENİ
        browser = await p.chromium.launch(
            channel="chrome",                # Gerçek Chrome binary = TLS parmakizi asıl Chrome ile eşleşir
            headless=(not args.headful),
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )

        # YENİ
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            locale="tr-TR",
            timezone_id="Europe/Istanbul",
            user_agent=HEADERS["User-Agent"],
        )
        context.set_default_timeout(30000)
        
        # Cloudflare'in ek JS sinyallerini kapat
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            Object.defineProperty(navigator, 'languages', { get: () => ['tr-TR', 'tr', 'en-US', 'en'] });
            window.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'permissions', {
                get: () => ({ query: () => Promise.resolve({ state: 'granted' }) })
            });
        """)
        
        page = await context.new_page()
        await stealth_async(page)   # no-op ama satırı silmene gerek yok

        try:
            await warmup_session(page, args.search_url)
        except Exception as e:
            log({"ts": now_iso(), "event": "warmup_failed", "error": str(e)})

        try:
            discovered = await harvest_listing_urls(page, args.search_url, max_pages=args.max_pages)
            discovered = [u for u in discovered if u not in visited]
            queue_set = set(queue)
            newly = [u for u in discovered if u not in queue_set]
            queue.extend(newly)
            write_lines(queue_path, queue)

            log({
                "ts": now_iso(),
                "event": "queue_filled",
                "discovered": len(discovered),
                "new_added": len(newly),
                "queue_size": len(queue),
            })
            print(f"[queue] discovered={len(discovered)} new_added={len(newly)} queue_size={len(queue)}")
        except Exception as e:
            log({"ts": now_iso(), "event": "queue_fill_failed", "error": str(e)})
            print(f"[queue] failed: {e}")

        processed = 0
        remaining_queue = []
        done_urls = []

        ml_ready_rows = []
        multimodal_ready_rows = []

        # batch buffers
        listings_buffer: List[Dict[str, Any]] = []
        images_buffer: List[Dict[str, Any]] = []
        validation_buffer: List[Dict[str, Any]] = []

        def flush_buffers():
            nonlocal listings_buffer, images_buffer, validation_buffer
            append_parquet(listings_path, listings_buffer, unique_key="listing_id")
            append_parquet(images_path, images_buffer, unique_key="image_id")
            append_parquet(validation_path, validation_buffer, unique_key="listing_id")
            listings_buffer = []
            images_buffer = []
            validation_buffer = []

        flush_every = max(1, args.flush_every)

        for url in queue:
            if processed >= args.limit:
                remaining_queue.append(url)
                continue

            if url in visited:
                continue

            try:
                listing_row, image_rows, validation_row = await parse_listing(
                    page=page,
                    url=url,
                    max_images=args.max_images,
                    max_next_clicks=args.max_next_clicks,
                    out_dir=out_dir,
                    resize_max_width=args.resize_max_width,
                    jpeg_quality=args.jpeg_quality,
                    min_img_width=args.min_img_width,
                    retry=args.retry,
                    delay_min=args.delay_min,
                    delay_max=args.delay_max,
                )

                listings_buffer.append(listing_row)
                images_buffer.extend(image_rows)
                validation_buffer.append(validation_row)

                if validation_row["is_train_ready_ml"]:
                    ml_ready_rows.append(listing_row)

                if validation_row["is_train_ready_multimodal"]:
                    multimodal_ready_rows.append({
                        **listing_row,
                        "valid_image_paths": json.dumps(
                            [img["local_path"] for img in image_rows if img["is_valid"]],
                            ensure_ascii=False
                        ),
                        "valid_image_count": validation_row["valid_image_count"],
                    })

                done_urls.append(url)
                visited.add(url)
                processed += 1

                log({
                    "ts": now_iso(),
                    "event": "listing_ok",
                    "url": url,
                    "listing_id": listing_row["listing_id"],
                    "images": len(image_rows),
                    "ml_ready": validation_row["is_train_ready_ml"],
                    "multimodal_ready": validation_row["is_train_ready_multimodal"],
                    "reject_reason": validation_row["reject_reason"],
                })

                print(
                    f"Bitti ({processed}/{args.limit}): "
                    f"{listing_row['listing_id']} | "
                    f"ML={validation_row['is_train_ready_ml']} | "
                    f"MM={validation_row['is_train_ready_multimodal']} | "
                    f"reason={validation_row['reject_reason']}"
                )

                if processed % flush_every == 0:
                    flush_buffers()

                await human_pause(args.delay_min, args.delay_max)

            except Exception as e:
                remaining_queue.append(url)
                log({"ts": now_iso(), "event": "listing_fail", "url": url, "error": str(e)})
                print(f"[fail] {url} -> {e}")

                try:
                    await page.goto(args.search_url, wait_until="domcontentloaded", timeout=60000)
                    await human_pause(0.8, 1.4)
                except Exception:
                    pass

        # last flush
        flush_buffers()

        await browser.close()

    append_lines(visited_path, done_urls)
    write_lines(queue_path, remaining_queue)

    append_parquet(train_ready_ml_path, ml_ready_rows, unique_key="listing_id")
    append_parquet(train_ready_multimodal_path, multimodal_ready_rows, unique_key="listing_id")

    log({
        "ts": now_iso(),
        "event": "run_end",
        "processed": processed,
        "queue_left": len(remaining_queue),
        "visited_total": len(visited),
        "ml_ready_added": len(ml_ready_rows),
        "multimodal_ready_added": len(multimodal_ready_rows),
    })

    print("Run complete.")
    print(f"Processed: {processed}")
    print(f"ML-ready added: {len(ml_ready_rows)}")
    print(f"Multimodal-ready added: {len(multimodal_ready_rows)}")
    print(f"Queue left: {len(remaining_queue)}")


def build_parser():
    ap = argparse.ArgumentParser(description="Hepsiemlak Ankara kiralik data agent.")
    ap.add_argument("--search-url", default=DEFAULT_SEARCH_URL)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--max-pages", type=int, default=60)
    ap.add_argument("--max-images", type=int, default=20)
    ap.add_argument("--max-next-clicks", type=int, default=40)
    ap.add_argument("--headful", action="store_true")
    ap.add_argument("--out-dir", default="dataset")
    ap.add_argument("--state-dir", default="data/state")
    ap.add_argument("--resize-max-width", type=int, default=1280)
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--min-img-width", type=int, default=320)
    ap.add_argument("--retry", type=int, default=2)
    ap.add_argument("--delay-min", type=float, default=0.4)
    ap.add_argument("--delay-max", type=float, default=0.9)
    ap.add_argument("--flush-every", type=int, default=25)
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    asyncio.run(run(args))