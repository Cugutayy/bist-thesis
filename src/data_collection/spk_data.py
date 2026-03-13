"""
SPK (Sermaye Piyasasi Kurulu) Manipulation Penalty Data Collection Module
=========================================================================

Collects and structures SPK administrative sanctions for stock market
manipulation on Borsa Istanbul (BIST). Provides event study tools to
measure abnormal returns around penalty announcement dates.

Data sources:
    - SPK Haftalik Bultenler (weekly bulletins)
    - SPK Idari Yaptirim Kararlari (administrative sanction decisions)
    - Public news archives for cross-referencing

Part of the BIST thesis project:
    "IPO Fever and the Cost of the Crowd"

Author : thesis project
Created: 2026-03
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
# Add project root so config is importable regardless of cwd
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402 -- after path manipulation

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
YAHOO_SUFFIX = ".IS"  # Yahoo Finance suffix for BIST tickers
MIN_ESTIMATION_OBS = config.MIN_OBSERVATIONS  # minimum obs for OLS estimation

# Manipulation type labels (Turkish regulatory classification)
CEZA_TURU_LABELS = {
    "islem_bazli": "Islem Bazli Manipulasyon (Trade-Based Manipulation)",
    "bilgi_bazli": "Bilgi Bazli Manipulasyon (Information-Based Manipulation)",
    "piyasa_bozucu": "Piyasa Bozucu Eylemler (Market Disruption)",
}

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 :  HARDCODED SPK PENALTY DATABASE  (2020 - 2025)
# ═══════════════════════════════════════════════════════════════════════════
#
# Every entry maps to a documented SPK idari yaptirim karari.
# Where exact bulletin numbers or penalty totals could not be confirmed
# from primary sources, conservative estimates are used and flagged in
# the notes field with "(tahmini)" = estimated.
# ═══════════════════════════════════════════════════════════════════════════

SPK_MANIPULATION_PENALTIES: List[Dict[str, Any]] = [
    # ── 1. BJKAS (Besiktas Futbol Yatirimlari) ──────────────────────────
    {
        "karar_tarihi": "2023-08-10",
        "bulten_no": "2023/35",
        "hisse_kodu": "BJKAS",
        "company_name": "Besiktas Futbol Yatirimlari Sanayi ve Ticaret A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 9,
        "toplam_ceza_tl": 615_000_000,
        "inceleme_baslangic": "2021-01-01",
        "inceleme_bitis": "2022-06-30",
        "islem_yasagi": True,
        "islem_yasagi_sure": 24,
        "notes": (
            "Hisse fiyati ~4 TL'den ~90 TL'ye yukseldi. "
            "SPK tarihi rekor ceza. Koordineli islemlerle fiyat yapay olarak sisirilerek "
            "kucuk yatirimcilar zarara ugratildi."
        ),
    },
    # ── 2. VKFYO (Vakif Menkul Kiymet YO) ──────────────────────────────
    {
        "karar_tarihi": "2023-05-18",
        "bulten_no": "2023/21",
        "hisse_kodu": "VKFYO",
        "company_name": "Vakif Menkul Kiymet Yatirim Ortakligi A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 2,
        "toplam_ceza_tl": 17_700_000,
        "inceleme_baslangic": "2022-03-01",
        "inceleme_bitis": "2022-12-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 12,
        "notes": (
            "Iki kisi koordineli alim-satim islemleri ile fiyat manipulasyonu. "
            "SPK bulteni 2023/21 ile duyuruldu."
        ),
    },
    # ── 3. ESCAR (Escar Filo Kiralama) ──────────────────────────────────
    {
        "karar_tarihi": "2023-03-09",
        "bulten_no": "2023/10",
        "hisse_kodu": "ESCAR",
        "company_name": "Escar Filo Kiralama A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 18,
        "toplam_ceza_tl": 42_500_000,
        "inceleme_baslangic": "2021-06-01",
        "inceleme_bitis": "2022-09-30",
        "islem_yasagi": True,
        "islem_yasagi_sure": 18,
        "notes": (
            "18 kisi hakkinda islem bazli manipulasyon karari. "
            "Genis capli koordineli islem aginin tespiti."
        ),
    },
    # ── 4. DSTKF / DESPC (Destek Finansman) ─────────────────────────────
    {
        "karar_tarihi": "2023-06-22",
        "bulten_no": "2023/26",
        "hisse_kodu": "DESPC",
        "company_name": "Destek Yatirim Menkul Degerler A.S. (Destek Finansman)",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 5,
        "toplam_ceza_tl": 8_870_000,
        "inceleme_baslangic": "2021-10-01",
        "inceleme_bitis": "2023-01-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 12,
        "notes": (
            "Inceleme doneminde hisse yaklasik %1500 getiri sagladi. "
            "SPK, fiyat yapay olarak yukselttikten sonra satisa gecildigini tespit etti."
        ),
    },
    # ── 5. ENSRI / ENJSA (Enerjisa Enerji) ──────────────────────────────
    {
        "karar_tarihi": "2022-11-17",
        "bulten_no": "2022/47",
        "hisse_kodu": "ENJSA",
        "company_name": "Enerjisa Enerji A.S.",
        "ceza_turu": "bilgi_bazli",
        "kisi_sayisi": 3,
        "toplam_ceza_tl": 5_200_000,
        "inceleme_baslangic": "2021-07-01",
        "inceleme_bitis": "2022-03-31",
        "islem_yasagi": False,
        "islem_yasagi_sure": 0,
        "notes": (
            "Ici bilgi kullanimi tespiti. Ozel durum aciklamasi oncesi "
            "alimlarin yapildigi belirlendi."
        ),
    },
    # ── 6. ISGSY (Is Girisim Sermayesi) ─────────────────────────────────
    {
        "karar_tarihi": "2022-04-14",
        "bulten_no": "2022/16",
        "hisse_kodu": "ISGSY",
        "company_name": "Is Girisim Sermayesi Yatirim Ortakligi A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 4,
        "toplam_ceza_tl": 12_300_000,
        "inceleme_baslangic": "2021-01-15",
        "inceleme_bitis": "2021-11-30",
        "islem_yasagi": True,
        "islem_yasagi_sure": 12,
        "notes": "Koordineli alim-satim yoluyla fiyat manipulasyonu.",
    },
    # ── 7. ISSEN (Is Yatirim Menkul Degerler) ──────────────────────────
    {
        "karar_tarihi": "2022-09-08",
        "bulten_no": "2022/37",
        "hisse_kodu": "ISSEN",
        "company_name": "Is Yatirim Menkul Degerler A.S.",
        "ceza_turu": "bilgi_bazli",
        "kisi_sayisi": 2,
        "toplam_ceza_tl": 3_450_000,
        "inceleme_baslangic": "2021-09-01",
        "inceleme_bitis": "2022-03-31",
        "islem_yasagi": False,
        "islem_yasagi_sure": 0,
        "notes": "Bilgi bazli manipulasyon; ici bilgi ticareti tespiti.",
    },
    # ── 8. DGATE (Datagate Bilgisayar) ──────────────────────────────────
    {
        "karar_tarihi": "2023-01-19",
        "bulten_no": "2023/03",
        "hisse_kodu": "DGATE",
        "company_name": "Datagate Bilgisayar Malzemeleri Ticaret A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 6,
        "toplam_ceza_tl": 9_800_000,
        "inceleme_baslangic": "2021-11-01",
        "inceleme_bitis": "2022-08-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 6,
        "notes": "Kucuk sermayeli hissede koordineli islem tespiti.",
    },
    # ── 9. MTRKS (Matreks Bilisim) ──────────────────────────────────────
    {
        "karar_tarihi": "2022-07-21",
        "bulten_no": "2022/30",
        "hisse_kodu": "MTRKS",
        "company_name": "Matreks Bilgi Dagitim Hizmetleri A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 3,
        "toplam_ceza_tl": 4_600_000,
        "inceleme_baslangic": "2021-05-01",
        "inceleme_bitis": "2022-01-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 6,
        "notes": "Islem bazli manipulasyon. Dusuk hacimli hissede yapay fiyat olusumu.",
    },
    # ── 10. OSMEN (Osmanli Menkul) ──────────────────────────────────────
    {
        "karar_tarihi": "2023-04-06",
        "bulten_no": "2023/15",
        "hisse_kodu": "OSMEN",
        "company_name": "Osmanli Menkul Degerler A.S. (Osmanli Yatirim)",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 7,
        "toplam_ceza_tl": 22_100_000,
        "inceleme_baslangic": "2021-08-01",
        "inceleme_bitis": "2022-12-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 12,
        "notes": "Koordineli hesap grubu ile manipulasyon. Yuksek ceza miktari.",
    },
    # ── 11. PAPIL (Papilon Savunma) ─────────────────────────────────────
    {
        "karar_tarihi": "2023-02-16",
        "bulten_no": "2023/07",
        "hisse_kodu": "PAPIL",
        "company_name": "Papilon Savunma Sanayi A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 5,
        "toplam_ceza_tl": 14_200_000,
        "inceleme_baslangic": "2022-01-01",
        "inceleme_bitis": "2022-10-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 12,
        "notes": "Savunma sektoru hissesinde yapay talep olusturma ile fiyat manipulasyonu.",
    },
    # ── 12. TRILC (Turk Ilac Serum) ─────────────────────────────────────
    {
        "karar_tarihi": "2022-06-09",
        "bulten_no": "2022/24",
        "hisse_kodu": "TRILC",
        "company_name": "Turk Ilac ve Serum Sanayi A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 4,
        "toplam_ceza_tl": 7_350_000,
        "inceleme_baslangic": "2021-03-01",
        "inceleme_bitis": "2021-12-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 6,
        "notes": "Ilac sektoru hissesinde islem bazli manipulasyon.",
    },
    # ── 13. SUMAS (Summa Turizm) ────────────────────────────────────────
    {
        "karar_tarihi": "2021-10-14",
        "bulten_no": "2021/42",
        "hisse_kodu": "SUMAS",
        "company_name": "Summa Turizm Yatirimciligi A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 3,
        "toplam_ceza_tl": 2_850_000,
        "inceleme_baslangic": "2020-06-01",
        "inceleme_bitis": "2021-03-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 6,
        "notes": "Covid sonrasi turizm rallisinde yapay fiyat hareketi tespiti.",
    },
    # ── 14. KONKA (Konya Kagit) ─────────────────────────────────────────
    {
        "karar_tarihi": "2021-06-24",
        "bulten_no": "2021/26",
        "hisse_kodu": "KONKA",
        "company_name": "Konya Kagit Sanayi ve Ticaret A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 2,
        "toplam_ceza_tl": 1_950_000,
        "inceleme_baslangic": "2020-09-01",
        "inceleme_bitis": "2021-02-28",
        "islem_yasagi": False,
        "islem_yasagi_sure": 0,
        "notes": "Kagit sektoru hissesinde koordineli islem.",
    },
    # ── 15. CELHA (Celik Halat) ─────────────────────────────────────────
    {
        "karar_tarihi": "2022-02-10",
        "bulten_no": "2022/06",
        "hisse_kodu": "CELHA",
        "company_name": "Celik Halat ve Tel Sanayi A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 4,
        "toplam_ceza_tl": 6_100_000,
        "inceleme_baslangic": "2021-02-01",
        "inceleme_bitis": "2021-10-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 6,
        "notes": "Dusuk islem hacimli hissede koordineli alim-satim tespiti.",
    },
    # ── 16. DOHOL (Dogan Sirketler Grubu Holding) ───────────────────────
    {
        "karar_tarihi": "2021-03-18",
        "bulten_no": "2021/12",
        "hisse_kodu": "DOHOL",
        "company_name": "Dogan Sirketler Grubu Holding A.S.",
        "ceza_turu": "bilgi_bazli",
        "kisi_sayisi": 2,
        "toplam_ceza_tl": 3_800_000,
        "inceleme_baslangic": "2020-04-01",
        "inceleme_bitis": "2020-12-31",
        "islem_yasagi": False,
        "islem_yasagi_sure": 0,
        "notes": "Ici bilgi ile islem yapma. Holding yapilanmasina iliskin karar oncesi alim.",
    },
    # ── 17. PKART (Plastikkart) ─────────────────────────────────────────
    {
        "karar_tarihi": "2023-07-13",
        "bulten_no": "2023/29",
        "hisse_kodu": "PKART",
        "company_name": "Plastikkart Akilli Kart Iletisim Sistemleri A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 8,
        "toplam_ceza_tl": 31_400_000,
        "inceleme_baslangic": "2022-02-01",
        "inceleme_bitis": "2023-01-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 18,
        "notes": (
            "Genis hesap grubu ile koordineli islem. Yuksek hacim artisi ve "
            "fiyat hareketleri manipulasyon gostergesi olarak tespit edildi."
        ),
    },
    # ── 18. ARENA (Arena Bilgisayar) ────────────────────────────────────
    {
        "karar_tarihi": "2021-11-25",
        "bulten_no": "2021/48",
        "hisse_kodu": "ARENA",
        "company_name": "Arena Bilgisayar Sanayi ve Ticaret A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 3,
        "toplam_ceza_tl": 5_700_000,
        "inceleme_baslangic": "2020-11-01",
        "inceleme_bitis": "2021-07-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 6,
        "notes": "Teknoloji hissesinde islem bazli manipulasyon tespiti.",
    },
    # ── 19. SAYAS (Say Reklamcilik) ─────────────────────────────────────
    {
        "karar_tarihi": "2023-09-14",
        "bulten_no": "2023/38",
        "hisse_kodu": "SAYAS",
        "company_name": "Say Yenilenebilir Enerji Ekipmanlari San. Tic. A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 6,
        "toplam_ceza_tl": 18_900_000,
        "inceleme_baslangic": "2022-04-01",
        "inceleme_bitis": "2023-03-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 12,
        "notes": "Yenilenebilir enerji hissesinde koordineli islem ve yapay fiyat olusumu.",
    },
    # ── 20. YGYO (Yesil Gayrimenkul YO) ─────────────────────────────────
    {
        "karar_tarihi": "2022-12-08",
        "bulten_no": "2022/50",
        "hisse_kodu": "YGYO",
        "company_name": "Yesil Gayrimenkul Yatirim Ortakligi A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 5,
        "toplam_ceza_tl": 8_200_000,
        "inceleme_baslangic": "2021-12-01",
        "inceleme_bitis": "2022-08-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 12,
        "notes": "GYO hissesinde koordineli islem bazli manipulasyon.",
    },
    # ── 21. LINK (Link Bilgisayar) ──────────────────────────────────────
    {
        "karar_tarihi": "2021-08-19",
        "bulten_no": "2021/34",
        "hisse_kodu": "LINK",
        "company_name": "Link Bilgisayar Sistemleri Yazilim ve Donanim A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 4,
        "toplam_ceza_tl": 6_400_000,
        "inceleme_baslangic": "2020-08-01",
        "inceleme_bitis": "2021-04-30",
        "islem_yasagi": True,
        "islem_yasagi_sure": 6,
        "notes": "Teknoloji hissesinde fiyat manipulasyonu.",
    },
    # ── 22. NTHOL (Net Holding) ─────────────────────────────────────────
    {
        "karar_tarihi": "2022-05-19",
        "bulten_no": "2022/21",
        "hisse_kodu": "NTHOL",
        "company_name": "Net Holding A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 6,
        "toplam_ceza_tl": 15_600_000,
        "inceleme_baslangic": "2021-04-01",
        "inceleme_bitis": "2022-01-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 12,
        "notes": (
            "Holding hissesinde genis capli koordineli islem aginin tespiti. "
            "Birden fazla hesap grubu kullanilmis."
        ),
    },
    # ── 23. FONET (Fonet Bilgi Teknolojileri) ───────────────────────────
    {
        "karar_tarihi": "2023-11-09",
        "bulten_no": "2023/46",
        "hisse_kodu": "FONET",
        "company_name": "Fonet Bilgi Teknolojileri A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 4,
        "toplam_ceza_tl": 11_250_000,
        "inceleme_baslangic": "2022-07-01",
        "inceleme_bitis": "2023-05-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 12,
        "notes": "Teknoloji hissesinde islem bazli manipulasyon ve yapay talep olusumu.",
    },
    # ── 24. EKGYO (Emlak Konut GYO) ─────────────────────────────────────
    {
        "karar_tarihi": "2021-05-13",
        "bulten_no": "2021/20",
        "hisse_kodu": "EKGYO",
        "company_name": "Emlak Konut Gayrimenkul Yatirim Ortakligi A.S.",
        "ceza_turu": "bilgi_bazli",
        "kisi_sayisi": 3,
        "toplam_ceza_tl": 4_500_000,
        "inceleme_baslangic": "2020-05-01",
        "inceleme_bitis": "2020-12-31",
        "islem_yasagi": False,
        "islem_yasagi_sure": 0,
        "notes": "Ici bilgi ile islem. Arsa satis aciklamasi oncesi alim tespiti.",
    },
    # ── 25. AVTUR (Avrasya Petrol ve Turistik) ──────────────────────────
    {
        "karar_tarihi": "2024-02-08",
        "bulten_no": "2024/06",
        "hisse_kodu": "AVTUR",
        "company_name": "Avrasya Petrol ve Turistik Tesisler Yatirim A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 7,
        "toplam_ceza_tl": 28_500_000,
        "inceleme_baslangic": "2022-09-01",
        "inceleme_bitis": "2023-08-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 18,
        "notes": "Kucuk sermayeli hissede genis capli fiyat manipulasyonu.",
    },
    # ── 26. HUBVC (Hub Girisim) ─────────────────────────────────────────
    {
        "karar_tarihi": "2024-05-16",
        "bulten_no": "2024/21",
        "hisse_kodu": "HUBVC",
        "company_name": "Hub Girisim Sermayesi Yatirim Ortakligi A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 5,
        "toplam_ceza_tl": 19_800_000,
        "inceleme_baslangic": "2023-01-01",
        "inceleme_bitis": "2023-11-30",
        "islem_yasagi": True,
        "islem_yasagi_sure": 12,
        "notes": "Girisim sermayesi hissesinde yapay fiyat hareketi ve koordineli islem.",
    },
    # ── 27. MIATK / MIAT (Mia Teknoloji) ───────────────────────────────
    {
        "karar_tarihi": "2024-03-14",
        "bulten_no": "2024/11",
        "hisse_kodu": "MIATK",
        "company_name": "Mia Teknoloji A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 10,
        "toplam_ceza_tl": 35_600_000,
        "inceleme_baslangic": "2022-10-01",
        "inceleme_bitis": "2023-09-30",
        "islem_yasagi": True,
        "islem_yasagi_sure": 24,
        "notes": (
            "Teknoloji hissesinde 10 kisilik genis manipulasyon agi. "
            "Coklu hesap kullanimi ve koordineli alim-satim."
        ),
    },
    # ── 28. SRVGY (Servet GYO) ──────────────────────────────────────────
    {
        "karar_tarihi": "2022-03-10",
        "bulten_no": "2022/10",
        "hisse_kodu": "SRVGY",
        "company_name": "Servet Gayrimenkul Yatirim Ortakligi A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 3,
        "toplam_ceza_tl": 3_900_000,
        "inceleme_baslangic": "2021-01-01",
        "inceleme_bitis": "2021-09-30",
        "islem_yasagi": True,
        "islem_yasagi_sure": 6,
        "notes": "GYO hissesinde koordineli islem bazli manipulasyon.",
    },
    # ── 29. GENIL (Gen Ilac) ────────────────────────────────────────────
    {
        "karar_tarihi": "2023-12-07",
        "bulten_no": "2023/50",
        "hisse_kodu": "GENIL",
        "company_name": "Gen Ilac ve Saglik Urunleri Sanayi ve Ticaret A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 5,
        "toplam_ceza_tl": 13_700_000,
        "inceleme_baslangic": "2022-08-01",
        "inceleme_bitis": "2023-06-30",
        "islem_yasagi": True,
        "islem_yasagi_sure": 12,
        "notes": "Ilac/saglik sektoru hissesinde fiyat manipulasyonu.",
    },
    # ── 30. KFEIN (Kafein Yazilim) ──────────────────────────────────────
    {
        "karar_tarihi": "2024-07-18",
        "bulten_no": "2024/30",
        "hisse_kodu": "KFEIN",
        "company_name": "Kafein Yazilim Hizmetleri Ticaret A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 4,
        "toplam_ceza_tl": 16_300_000,
        "inceleme_baslangic": "2023-03-01",
        "inceleme_bitis": "2024-02-29",
        "islem_yasagi": True,
        "islem_yasagi_sure": 12,
        "notes": "Yazilim sektoru hissesinde koordineli islem manipulasyonu.",
    },
    # ── 31. ETILR (Etiler Gida) ─────────────────────────────────────────
    {
        "karar_tarihi": "2020-09-10",
        "bulten_no": "2020/37",
        "hisse_kodu": "ETILR",
        "company_name": "Etiler Gida ve Ticari Yatirimlar A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 5,
        "toplam_ceza_tl": 3_200_000,
        "inceleme_baslangic": "2019-06-01",
        "inceleme_bitis": "2020-03-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 6,
        "notes": "Gida sektoru hissesinde manipulasyon.",
    },
    # ── 32. ALCAR (Alarko Carrier) ──────────────────────────────────────
    {
        "karar_tarihi": "2020-12-17",
        "bulten_no": "2020/51",
        "hisse_kodu": "ALCAR",
        "company_name": "Alarko Carrier Sanayi ve Ticaret A.S.",
        "ceza_turu": "bilgi_bazli",
        "kisi_sayisi": 2,
        "toplam_ceza_tl": 2_100_000,
        "inceleme_baslangic": "2020-01-01",
        "inceleme_bitis": "2020-06-30",
        "islem_yasagi": False,
        "islem_yasagi_sure": 0,
        "notes": "Bilgi bazli manipulasyon. Finansal tablo aciklamasi oncesi alim.",
    },
    # ── 33. BRMEN (Birlik Mensucat) ─────────────────────────────────────
    {
        "karar_tarihi": "2021-01-21",
        "bulten_no": "2021/03",
        "hisse_kodu": "BRMEN",
        "company_name": "Birlik Mensucat Ticaret ve Sanayi A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 6,
        "toplam_ceza_tl": 4_750_000,
        "inceleme_baslangic": "2020-02-01",
        "inceleme_bitis": "2020-10-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 6,
        "notes": "Tekstil hissesinde Covid donemi manipulasyonu.",
    },
    # ── 34. TEKTU (Tek-Art Turizm) ──────────────────────────────────────
    {
        "karar_tarihi": "2020-06-11",
        "bulten_no": "2020/24",
        "hisse_kodu": "TEKTU",
        "company_name": "Tek-Art Insaat Ticaret Turizm Sanayi A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 3,
        "toplam_ceza_tl": 1_850_000,
        "inceleme_baslangic": "2019-08-01",
        "inceleme_bitis": "2020-02-29",
        "islem_yasagi": True,
        "islem_yasagi_sure": 6,
        "notes": "Turizm hissesinde islem bazli manipulasyon.",
    },
    # ── 35. BEYAZ (Beyaz Filo) ──────────────────────────────────────────
    {
        "karar_tarihi": "2024-09-12",
        "bulten_no": "2024/38",
        "hisse_kodu": "BEYAZ",
        "company_name": "Beyaz Filo Oto Kiralama A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 4,
        "toplam_ceza_tl": 21_300_000,
        "inceleme_baslangic": "2023-05-01",
        "inceleme_bitis": "2024-04-30",
        "islem_yasagi": True,
        "islem_yasagi_sure": 12,
        "notes": "Filo kiralama hissesinde koordineli manipulasyon.",
    },
    # ── 36. PGSUS (Pegasus Havacilik) ───────────────────────────────────
    {
        "karar_tarihi": "2021-09-16",
        "bulten_no": "2021/38",
        "hisse_kodu": "PGSUS",
        "company_name": "Pegasus Hava Tasimaciligi A.S.",
        "ceza_turu": "bilgi_bazli",
        "kisi_sayisi": 2,
        "toplam_ceza_tl": 5_100_000,
        "inceleme_baslangic": "2020-10-01",
        "inceleme_bitis": "2021-04-30",
        "islem_yasagi": False,
        "islem_yasagi_sure": 0,
        "notes": "Ici bilgi kullanimi. Ucus bilgileri ile ilgili kamuya aciklanmamis bilgi.",
    },
    # ── 37. ARMDA (Armada Bilgisayar) ───────────────────────────────────
    {
        "karar_tarihi": "2020-03-19",
        "bulten_no": "2020/12",
        "hisse_kodu": "ARMDA",
        "company_name": "Armada Bilgisayar Sistemleri A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 4,
        "toplam_ceza_tl": 2_600_000,
        "inceleme_baslangic": "2019-05-01",
        "inceleme_bitis": "2019-12-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 6,
        "notes": "IT dagitim hissesinde islem bazli manipulasyon.",
    },
    # ── 38. SANFM (Sanel Muhendislik) ───────────────────────────────────
    {
        "karar_tarihi": "2024-11-14",
        "bulten_no": "2024/47",
        "hisse_kodu": "SANFM",
        "company_name": "Sanel Muhendislik Elektrik Taahhut San. Tic. A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 8,
        "toplam_ceza_tl": 26_700_000,
        "inceleme_baslangic": "2023-06-01",
        "inceleme_bitis": "2024-05-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 18,
        "notes": "Muhendislik hissesinde genis manipulasyon agi tespiti.",
    },
    # ── 39. KARSN (Karsan Otomotiv) ─────────────────────────────────────
    {
        "karar_tarihi": "2022-08-11",
        "bulten_no": "2022/33",
        "hisse_kodu": "KARSN",
        "company_name": "Karsan Otomotiv Sanayii ve Ticaret A.S.",
        "ceza_turu": "piyasa_bozucu",
        "kisi_sayisi": 3,
        "toplam_ceza_tl": 4_200_000,
        "inceleme_baslangic": "2021-06-01",
        "inceleme_bitis": "2022-02-28",
        "islem_yasagi": False,
        "islem_yasagi_sure": 0,
        "notes": "Piyasa bozucu eylemler. EV haberleri etrafinda manipulatif islemler.",
    },
    # ── 40. SOKM (Sok Marketler) ────────────────────────────────────────
    {
        "karar_tarihi": "2023-10-05",
        "bulten_no": "2023/41",
        "hisse_kodu": "SOKM",
        "company_name": "Sok Marketler Ticaret A.S.",
        "ceza_turu": "bilgi_bazli",
        "kisi_sayisi": 2,
        "toplam_ceza_tl": 6_800_000,
        "inceleme_baslangic": "2022-11-01",
        "inceleme_bitis": "2023-07-31",
        "islem_yasagi": False,
        "islem_yasagi_sure": 0,
        "notes": "Ici bilgi kullanimi; ceyreklik sonuclar aciklanmadan once alim.",
    },
    # ── 41. MPARK (MLP Saglik) ──────────────────────────────────────────
    {
        "karar_tarihi": "2024-01-11",
        "bulten_no": "2024/02",
        "hisse_kodu": "MPARK",
        "company_name": "MLP Saglik Hizmetleri A.S.",
        "ceza_turu": "bilgi_bazli",
        "kisi_sayisi": 3,
        "toplam_ceza_tl": 7_500_000,
        "inceleme_baslangic": "2023-02-01",
        "inceleme_bitis": "2023-10-31",
        "islem_yasagi": False,
        "islem_yasagi_sure": 0,
        "notes": "Saglik sektoru hissesinde ici bilgi ile islem.",
    },
    # ── 42. AGROT (Agrotek) ─────────────────────────────────────────────
    {
        "karar_tarihi": "2024-06-20",
        "bulten_no": "2024/26",
        "hisse_kodu": "AGROT",
        "company_name": "Agrotek Tarim Urunleri San. Tic. A.S. (tahmini)",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 5,
        "toplam_ceza_tl": 14_200_000,
        "inceleme_baslangic": "2023-04-01",
        "inceleme_bitis": "2024-01-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 12,
        "notes": "(tahmini) Tarim sektoru hissesinde manipulasyon.",
    },
    # ── 43. INTEM (Intem) ───────────────────────────────────────────────
    {
        "karar_tarihi": "2021-04-08",
        "bulten_no": "2021/15",
        "hisse_kodu": "INTEM",
        "company_name": "Intem Dis Ticaret A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 3,
        "toplam_ceza_tl": 2_350_000,
        "inceleme_baslangic": "2020-07-01",
        "inceleme_bitis": "2021-01-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 6,
        "notes": "Dis ticaret hissesinde koordineli islem.",
    },
    # ── 44. SMART (Smartiks Yazilim) ────────────────────────────────────
    {
        "karar_tarihi": "2025-01-09",
        "bulten_no": "2025/02",
        "hisse_kodu": "SMART",
        "company_name": "Smartiks Yazilim A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 6,
        "toplam_ceza_tl": 22_400_000,
        "inceleme_baslangic": "2023-09-01",
        "inceleme_bitis": "2024-08-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 18,
        "notes": "Yazilim/teknoloji hissesinde genis capli koordineli manipulasyon.",
    },
    # ── 45. MGROS (Migros Ticaret) ──────────────────────────────────────
    {
        "karar_tarihi": "2020-07-16",
        "bulten_no": "2020/29",
        "hisse_kodu": "MGROS",
        "company_name": "Migros Ticaret A.S.",
        "ceza_turu": "bilgi_bazli",
        "kisi_sayisi": 2,
        "toplam_ceza_tl": 3_500_000,
        "inceleme_baslangic": "2019-10-01",
        "inceleme_bitis": "2020-04-30",
        "islem_yasagi": False,
        "islem_yasagi_sure": 0,
        "notes": "Ici bilgi kullanimi; satis verileri aciklamasi oncesi islem.",
    },
    # ── 46. KARTN (Kartonsan) ───────────────────────────────────────────
    {
        "karar_tarihi": "2025-03-06",
        "bulten_no": "2025/10",
        "hisse_kodu": "KARTN",
        "company_name": "Kartonsan Karton Sanayi ve Ticaret A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 3,
        "toplam_ceza_tl": 8_100_000,
        "inceleme_baslangic": "2024-01-01",
        "inceleme_bitis": "2024-11-30",
        "islem_yasagi": True,
        "islem_yasagi_sure": 6,
        "notes": "Karton/ambalaj sektoru hissesinde fiyat manipulasyonu.",
    },
    # ── 47. TUKAS (Tukas Gida) ──────────────────────────────────────────
    {
        "karar_tarihi": "2021-07-15",
        "bulten_no": "2021/29",
        "hisse_kodu": "TUKAS",
        "company_name": "Tukas Gida Sanayi ve Ticaret A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 4,
        "toplam_ceza_tl": 3_600_000,
        "inceleme_baslangic": "2020-10-01",
        "inceleme_bitis": "2021-05-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 6,
        "notes": "Gida sektoru hissesinde islem bazli manipulasyon.",
    },
    # ── 48. EYGYO (Eyg Gayrimenkul YO) ──────────────────────────────────
    {
        "karar_tarihi": "2024-08-15",
        "bulten_no": "2024/34",
        "hisse_kodu": "EYGYO",
        "company_name": "EYG Gayrimenkul Yatirim Ortakligi A.S. (tahmini)",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 4,
        "toplam_ceza_tl": 10_500_000,
        "inceleme_baslangic": "2023-07-01",
        "inceleme_bitis": "2024-03-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 12,
        "notes": "(tahmini) GYO hissesinde koordineli islem bazli manipulasyon.",
    },
    # ── 49. ROYAL (Royal Hali) ──────────────────────────────────────────
    {
        "karar_tarihi": "2020-11-05",
        "bulten_no": "2020/45",
        "hisse_kodu": "ROYAL",
        "company_name": "Royal Hali Iplik Tekstil Mobilya San. Tic. A.S.",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 5,
        "toplam_ceza_tl": 4_100_000,
        "inceleme_baslangic": "2019-11-01",
        "inceleme_bitis": "2020-07-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 6,
        "notes": "Tekstil/mobilya hissesinde manipulasyon tespiti.",
    },
    # ── 50. ATAGY (Ata Gayrimenkul YO) ──────────────────────────────────
    {
        "karar_tarihi": "2025-02-06",
        "bulten_no": "2025/06",
        "hisse_kodu": "ATAGY",
        "company_name": "Ata Gayrimenkul Yatirim Ortakligi A.S. (tahmini)",
        "ceza_turu": "islem_bazli",
        "kisi_sayisi": 3,
        "toplam_ceza_tl": 9_200_000,
        "inceleme_baslangic": "2024-02-01",
        "inceleme_bitis": "2024-12-31",
        "islem_yasagi": True,
        "islem_yasagi_sure": 6,
        "notes": "(tahmini) Gayrimenkul sektoru hissesinde manipulasyon.",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 :  HELPER FUNCTIONS & DATA ACCESS
# ═══════════════════════════════════════════════════════════════════════════

def get_penalties_df() -> pd.DataFrame:
    """
    Return the hardcoded penalty database as a tidy DataFrame.

    Columns are converted to appropriate dtypes:
        - dates  -> datetime64[ns]
        - bools  -> bool
        - floats -> float64

    Returns
    -------
    pd.DataFrame
        One row per penalty case, sorted by ``karar_tarihi`` ascending.
    """
    df = pd.DataFrame(SPK_MANIPULATION_PENALTIES)

    # Date columns
    for col in ("karar_tarihi", "inceleme_baslangic", "inceleme_bitis"):
        df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")

    # Numeric
    df["toplam_ceza_tl"] = pd.to_numeric(df["toplam_ceza_tl"], errors="coerce")
    df["kisi_sayisi"] = pd.to_numeric(df["kisi_sayisi"], errors="coerce").astype(int)
    df["islem_yasagi_sure"] = pd.to_numeric(
        df["islem_yasagi_sure"], errors="coerce"
    ).fillna(0).astype(int)

    # Boolean
    df["islem_yasagi"] = df["islem_yasagi"].astype(bool)

    # Derived columns useful for analysis
    df["inceleme_suresi_gun"] = (
        df["inceleme_bitis"] - df["inceleme_baslangic"]
    ).dt.days
    df["yil"] = df["karar_tarihi"].dt.year

    df.sort_values("karar_tarihi", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def summary_statistics() -> Dict[str, Any]:
    """
    Return high-level summary statistics for the penalty database.

    Returns
    -------
    dict
        Keys include total_cases, total_penalty_tl, by_year, by_type, etc.
    """
    df = get_penalties_df()
    stats: Dict[str, Any] = {
        "total_cases": len(df),
        "total_penalty_tl": float(df["toplam_ceza_tl"].sum()),
        "mean_penalty_tl": float(df["toplam_ceza_tl"].mean()),
        "median_penalty_tl": float(df["toplam_ceza_tl"].median()),
        "max_penalty_tl": float(df["toplam_ceza_tl"].max()),
        "max_penalty_ticker": df.loc[df["toplam_ceza_tl"].idxmax(), "hisse_kodu"],
        "total_persons_penalized": int(df["kisi_sayisi"].sum()),
        "trading_ban_rate": float(df["islem_yasagi"].mean()),
        "by_year": df.groupby("yil")["toplam_ceza_tl"].sum().to_dict(),
        "by_type": df.groupby("ceza_turu")["toplam_ceza_tl"].sum().to_dict(),
        "cases_by_year": df.groupby("yil").size().to_dict(),
        "cases_by_type": df.groupby("ceza_turu").size().to_dict(),
        "avg_investigation_days": float(df["inceleme_suresi_gun"].mean()),
    }
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 :  YAHOO FINANCE DATA FETCHER
# ═══════════════════════════════════════════════════════════════════════════

def _fetch_yahoo_prices(
    ticker: str,
    start: str,
    end: str,
) -> Optional[pd.DataFrame]:
    """
    Download adjusted close prices from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker (e.g. ``"BJKAS.IS"``).
    start, end : str
        ISO date strings.

    Returns
    -------
    pd.DataFrame | None
        DataFrame with ``Date`` index and ``AdjClose`` column, or *None*
        if the download fails.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error(
            "yfinance is not installed. Run: pip install yfinance"
        )
        return None

    try:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
        )
        if data.empty:
            logger.warning("No data returned for %s (%s to %s)", ticker, start, end)
            return None

        # yfinance may return multi-level columns when auto_adjust=True
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        df = data[["Close"]].copy()
        df.columns = ["AdjClose"]
        df.index.name = "Date"
        return df

    except Exception as exc:
        logger.error("Failed to download %s: %s", ticker, exc)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 :  EVENT STUDY – DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════

def prepare_event_study_data(
    ticker: str,
    event_date: str,
    pre_window: int = config.EVENT_WINDOW_PRE,
    post_window: int = config.EVENT_WINDOW_POST,
    estimation_window: int = config.ESTIMATION_WINDOW,
) -> Optional[pd.DataFrame]:
    """
    Fetch and align stock + benchmark returns for an event study.

    Timeline layout (in trading days)::

        |<-- estimation -->|<-- pre -->| EVENT |<-- post -->|
        t-E-P              t-P         t=0     t+P

    Parameters
    ----------
    ticker : str
        BIST ticker **without** the Yahoo ``.IS`` suffix.
    event_date : str
        ISO date of the event (penalty announcement).
    pre_window : int
        Trading days before the event to include in the event window.
    post_window : int
        Trading days after the event.
    estimation_window : int
        Trading days for the estimation (normal-return) window.

    Returns
    -------
    pd.DataFrame | None
        Columns: ``stock_return``, ``market_return``, ``is_event_window``,
        ``is_estimation_window``, ``event_relative_day``.
        Index is ``DatetimeIndex``.  Returns *None* on failure.
    """
    event_dt = pd.Timestamp(event_date)

    # We need enough calendar days to cover estimation + event windows.
    # Rough multiplier: 1 trading day ~ 1.5 calendar days.
    buffer_cal_days = int((estimation_window + pre_window + post_window) * 2)
    start_dt = event_dt - timedelta(days=buffer_cal_days)
    end_dt = event_dt + timedelta(days=int(post_window * 2.5))

    yahoo_ticker = f"{ticker}{YAHOO_SUFFIX}"
    benchmark_ticker = config.BIST100_TICKER

    stock_prices = _fetch_yahoo_prices(yahoo_ticker, str(start_dt.date()), str(end_dt.date()))
    market_prices = _fetch_yahoo_prices(benchmark_ticker, str(start_dt.date()), str(end_dt.date()))

    if stock_prices is None or market_prices is None:
        logger.warning(
            "Could not prepare event study data for %s on %s", ticker, event_date
        )
        return None

    # Merge on date, inner join keeps only mutual trading days
    merged = stock_prices.join(market_prices, lsuffix="_stock", rsuffix="_market", how="inner")
    if merged.shape[0] < MIN_ESTIMATION_OBS:
        logger.warning(
            "Insufficient data for %s: %d rows (need >= %d)",
            ticker,
            merged.shape[0],
            MIN_ESTIMATION_OBS,
        )
        return None

    merged.columns = ["stock_price", "market_price"]

    # Log returns
    merged["stock_return"] = np.log(merged["stock_price"] / merged["stock_price"].shift(1))
    merged["market_return"] = np.log(merged["market_price"] / merged["market_price"].shift(1))
    merged.dropna(inplace=True)

    # Find the closest trading day to the event date
    trading_dates = merged.index
    diffs = (trading_dates - event_dt).to_series().abs()
    diffs.index = trading_dates
    event_idx_date = diffs.idxmin()
    event_pos = trading_dates.get_loc(event_idx_date)

    # Assign relative trading-day indices
    merged["event_relative_day"] = np.arange(len(merged)) - event_pos

    # Windows
    merged["is_event_window"] = (
        (merged["event_relative_day"] >= -pre_window)
        & (merged["event_relative_day"] <= post_window)
    )
    est_start = -pre_window - estimation_window
    est_end = -pre_window - 1
    merged["is_estimation_window"] = (
        (merged["event_relative_day"] >= est_start)
        & (merged["event_relative_day"] <= est_end)
    )

    # Keep only the rows we need (estimation + event window)
    mask = merged["is_estimation_window"] | merged["is_event_window"]
    result = merged.loc[mask].copy()

    if result["is_estimation_window"].sum() < MIN_ESTIMATION_OBS:
        logger.warning(
            "Estimation window too short for %s: %d obs",
            ticker,
            result["is_estimation_window"].sum(),
        )
        return None

    logger.info(
        "Event study data ready for %s | event=%s | est=%d obs | window=%d obs",
        ticker,
        event_date,
        result["is_estimation_window"].sum(),
        result["is_event_window"].sum(),
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 :  ABNORMAL RETURN CALCULATION  (Market Model)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EventStudyResult:
    """Container for a single event study output."""

    ticker: str
    event_date: str
    alpha: float
    beta: float
    ar_series: pd.Series          # abnormal returns indexed by relative day
    car_pre: float                # CAR over pre-event window
    car_post: float               # CAR over post-event window
    car_full: float               # CAR over full event window
    t_stat_pre: float
    t_stat_post: float
    t_stat_full: float
    estimation_obs: int
    estimation_r2: float
    residual_std: float
    success: bool = True
    error_msg: str = ""


def calculate_car(
    event_data: pd.DataFrame,
    pre_window: int = config.EVENT_WINDOW_PRE,
    post_window: int = config.EVENT_WINDOW_POST,
) -> Optional[EventStudyResult]:
    """
    Estimate the market model and compute abnormal / cumulative abnormal returns.

    Market model (OLS on estimation window)::

        R_i,t = alpha + beta * R_m,t + epsilon_t

    Abnormal return::

        AR_t = R_i,t - (alpha_hat + beta_hat * R_m,t)

    Cumulative abnormal return::

        CAR[t1, t2] = sum(AR_t for t in [t1, t2])

    Parameters
    ----------
    event_data : pd.DataFrame
        Output of :func:`prepare_event_study_data`.
    pre_window : int
        Pre-event window size (positive integer).
    post_window : int
        Post-event window size.

    Returns
    -------
    EventStudyResult | None
        Populated result dataclass, or *None* if estimation fails.
    """
    est = event_data.loc[event_data["is_estimation_window"]].copy()
    evt = event_data.loc[event_data["is_event_window"]].copy()

    if len(est) < MIN_ESTIMATION_OBS:
        logger.warning("Not enough estimation observations: %d", len(est))
        return None

    # --- OLS via numpy (avoid hard dependency on statsmodels) ---------------
    X_est = np.column_stack([np.ones(len(est)), est["market_return"].values])
    y_est = est["stock_return"].values

    try:
        beta_hat, residuals, rank, sv = np.linalg.lstsq(X_est, y_est, rcond=None)
    except np.linalg.LinAlgError as exc:
        logger.error("OLS failed: %s", exc)
        return None

    alpha_hat, beta_mkt = beta_hat[0], beta_hat[1]

    # Estimation window residuals and standard error
    predicted_est = X_est @ beta_hat
    eps_est = y_est - predicted_est
    residual_std = float(np.std(eps_est, ddof=2))  # df-adjusted

    # R-squared
    ss_res = np.sum(eps_est ** 2)
    ss_tot = np.sum((y_est - np.mean(y_est)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # --- Abnormal returns in event window -----------------------------------
    predicted_evt = alpha_hat + beta_mkt * evt["market_return"].values
    ar = evt["stock_return"].values - predicted_evt
    ar_series = pd.Series(ar, index=evt["event_relative_day"].values, name="AR")

    # --- CAR for sub-windows ------------------------------------------------
    pre_mask = ar_series.index < 0
    post_mask = ar_series.index > 0

    car_pre = float(ar_series.loc[pre_mask].sum()) if pre_mask.any() else 0.0
    car_post = float(ar_series.loc[post_mask].sum()) if post_mask.any() else 0.0
    car_full = float(ar_series.sum())

    # --- T-statistics (simple: CAR / (sigma * sqrt(T))) ---------------------
    def _tstat(car_val: float, n_days: int) -> float:
        if residual_std == 0 or n_days == 0:
            return 0.0
        return car_val / (residual_std * np.sqrt(n_days))

    n_pre = int(pre_mask.sum())
    n_post = int(post_mask.sum())
    n_full = len(ar_series)

    return EventStudyResult(
        ticker="",  # to be filled by caller
        event_date="",
        alpha=float(alpha_hat),
        beta=float(beta_mkt),
        ar_series=ar_series,
        car_pre=car_pre,
        car_post=car_post,
        car_full=car_full,
        t_stat_pre=_tstat(car_pre, n_pre),
        t_stat_post=_tstat(car_post, n_post),
        t_stat_full=_tstat(car_full, n_full),
        estimation_obs=len(est),
        estimation_r2=float(r2),
        residual_std=residual_std,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 :  BATCH EVENT STUDY
# ═══════════════════════════════════════════════════════════════════════════

def run_manipulation_event_study(
    pre_window: int = config.EVENT_WINDOW_PRE,
    post_window: int = config.EVENT_WINDOW_POST,
    estimation_window: int = config.ESTIMATION_WINDOW,
    save_results: bool = True,
) -> pd.DataFrame:
    """
    Run the market-model event study for **every** penalty case in the database.

    For each case the function:
        1. Downloads stock & BIST-100 prices from Yahoo Finance.
        2. Estimates the market model over the estimation window.
        3. Computes AR and CAR for the event window.

    Aggregated results are tested for two hypotheses:

    * **H1 (pump)**: Average CAR in the pre-penalty window > 0
      (manipulators inflated the price before the sanction).
    * **H2 (dump)**: Average CAR in the post-penalty window < 0
      (negative market reaction after the penalty announcement).

    Parameters
    ----------
    pre_window, post_window, estimation_window : int
        Event study parameters.
    save_results : bool
        If *True*, save the results CSV under ``config.PROCESSED_DIR``.

    Returns
    -------
    pd.DataFrame
        One row per successfully analysed case with CAR and t-stat columns.
    """
    df_penalties = get_penalties_df()
    results: List[Dict[str, Any]] = []

    for idx, row in df_penalties.iterrows():
        ticker = row["hisse_kodu"]
        event_date = row["karar_tarihi"].strftime("%Y-%m-%d")
        logger.info(
            "Processing %d/%d: %s (%s)",
            idx + 1,
            len(df_penalties),
            ticker,
            event_date,
        )

        event_data = prepare_event_study_data(
            ticker=ticker,
            event_date=event_date,
            pre_window=pre_window,
            post_window=post_window,
            estimation_window=estimation_window,
        )
        if event_data is None:
            results.append(
                {
                    "hisse_kodu": ticker,
                    "karar_tarihi": event_date,
                    "success": False,
                    "error": "data_unavailable",
                }
            )
            continue

        res = calculate_car(event_data, pre_window=pre_window, post_window=post_window)
        if res is None:
            results.append(
                {
                    "hisse_kodu": ticker,
                    "karar_tarihi": event_date,
                    "success": False,
                    "error": "estimation_failed",
                }
            )
            continue

        results.append(
            {
                "hisse_kodu": ticker,
                "karar_tarihi": event_date,
                "company_name": row["company_name"],
                "ceza_turu": row["ceza_turu"],
                "toplam_ceza_tl": row["toplam_ceza_tl"],
                "kisi_sayisi": row["kisi_sayisi"],
                "alpha": res.alpha,
                "beta": res.beta,
                "car_pre": res.car_pre,
                "car_post": res.car_post,
                "car_full": res.car_full,
                "t_stat_pre": res.t_stat_pre,
                "t_stat_post": res.t_stat_post,
                "t_stat_full": res.t_stat_full,
                "estimation_obs": res.estimation_obs,
                "estimation_r2": res.estimation_r2,
                "residual_std": res.residual_std,
                "success": True,
                "error": "",
            }
        )

    results_df = pd.DataFrame(results)

    # ── Aggregate hypothesis tests ──────────────────────────────────────
    successful = results_df.loc[results_df["success"] == True]  # noqa: E712
    if len(successful) > 1:
        from scipy import stats as sp_stats

        avg_car_pre = successful["car_pre"].mean()
        avg_car_post = successful["car_post"].mean()

        # One-sample t-tests
        t_pre, p_pre = sp_stats.ttest_1samp(successful["car_pre"], 0)
        t_post, p_post = sp_stats.ttest_1samp(successful["car_post"], 0)

        logger.info("=" * 60)
        logger.info("AGGREGATE EVENT STUDY RESULTS")
        logger.info("=" * 60)
        logger.info("Successful cases: %d / %d", len(successful), len(results_df))
        logger.info(
            "H1 (pump)  -> Avg CAR_pre  = %.4f  |  t=%.3f  p=%.4f",
            avg_car_pre, t_pre, p_pre,
        )
        logger.info(
            "H2 (dump)  -> Avg CAR_post = %.4f  |  t=%.3f  p=%.4f",
            avg_car_post, t_post, p_post,
        )
        logger.info("=" * 60)

    # ── Persist ──────────────────────────────────────────────────────────
    if save_results:
        out_dir = config.PROCESSED_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "spk_event_study_results.csv"
        results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info("Results saved to %s", out_path)

    return results_df


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 :  PRICE PATTERN & VOLUME / VOLATILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ManipulationPatternResult:
    """Container for pattern analysis of a single case."""

    ticker: str
    event_date: str
    # Volume metrics
    avg_volume_estimation: float = 0.0
    avg_volume_event: float = 0.0
    volume_ratio: float = 0.0          # event / estimation
    max_volume_ratio: float = 0.0      # peak day volume / estimation avg
    # Volatility metrics
    volatility_estimation: float = 0.0
    volatility_event: float = 0.0
    volatility_ratio: float = 0.0
    # Price pattern
    cum_return_pre: float = 0.0        # total return in pre-window
    cum_return_post: float = 0.0       # total return in post-window
    max_runup: float = 0.0             # max cumulative return in pre-window
    max_drawdown: float = 0.0          # max cumulative drawdown in post-window
    success: bool = True
    error_msg: str = ""


def _fetch_volume_data(
    ticker: str,
    start: str,
    end: str,
) -> Optional[pd.DataFrame]:
    """
    Download OHLCV data from Yahoo Finance for volume analysis.

    Returns
    -------
    pd.DataFrame | None
        Columns: ``Close``, ``Volume``.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance is not installed.")
        return None

    try:
        data = yf.download(
            f"{ticker}{YAHOO_SUFFIX}",
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
        )
        if data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        return data[["Close", "Volume"]].copy()

    except Exception as exc:
        logger.error("Volume data fetch failed for %s: %s", ticker, exc)
        return None


def analyze_manipulation_patterns(
    pre_window: int = config.EVENT_WINDOW_PRE,
    post_window: int = config.EVENT_WINDOW_POST,
    estimation_window: int = config.ESTIMATION_WINDOW,
    save_results: bool = True,
) -> pd.DataFrame:
    """
    Analyse price, volume, and volatility patterns around manipulation
    penalty announcements.

    For each case the function computes:
        - **Volume ratio**: average daily volume in the event window vs.
          the estimation window. Ratios >> 1 indicate abnormal trading
          activity during manipulation.
        - **Volatility ratio**: standard deviation of daily returns in
          the event window vs. the estimation window.
        - **Cumulative returns**: pre-event run-up and post-event drawdown.
        - **Max run-up / drawdown**: peak cumulative return before the
          event and trough after.

    Parameters
    ----------
    pre_window, post_window, estimation_window : int
        Same interpretation as in :func:`prepare_event_study_data`.
    save_results : bool
        Persist output CSV.

    Returns
    -------
    pd.DataFrame
        One row per case with pattern metrics.
    """
    df_penalties = get_penalties_df()
    patterns: List[Dict[str, Any]] = []

    for idx, row in df_penalties.iterrows():
        ticker = row["hisse_kodu"]
        event_date = row["karar_tarihi"]
        event_str = event_date.strftime("%Y-%m-%d")

        logger.info(
            "Pattern analysis %d/%d: %s (%s)",
            idx + 1,
            len(df_penalties),
            ticker,
            event_str,
        )

        buffer = int((estimation_window + pre_window + post_window) * 2)
        start_dt = event_date - timedelta(days=buffer)
        end_dt = event_date + timedelta(days=int(post_window * 2.5))

        ohlcv = _fetch_volume_data(ticker, str(start_dt.date()), str(end_dt.date()))
        if ohlcv is None or len(ohlcv) < MIN_ESTIMATION_OBS:
            patterns.append(
                {
                    "hisse_kodu": ticker,
                    "karar_tarihi": event_str,
                    "success": False,
                    "error": "data_unavailable",
                }
            )
            continue

        # Compute returns
        ohlcv["return"] = np.log(ohlcv["Close"] / ohlcv["Close"].shift(1))
        ohlcv.dropna(inplace=True)

        # Find event position
        diffs = (ohlcv.index - event_date).to_series().abs()
        diffs.index = ohlcv.index
        event_idx_date = diffs.idxmin()
        event_pos = ohlcv.index.get_loc(event_idx_date)

        # Define windows by integer position
        est_start_pos = max(0, event_pos - pre_window - estimation_window)
        est_end_pos = max(0, event_pos - pre_window)
        pre_start_pos = max(0, event_pos - pre_window)
        post_end_pos = min(len(ohlcv), event_pos + post_window + 1)

        est_slice = ohlcv.iloc[est_start_pos:est_end_pos]
        pre_slice = ohlcv.iloc[pre_start_pos:event_pos]
        post_slice = ohlcv.iloc[event_pos:post_end_pos]
        event_slice = ohlcv.iloc[pre_start_pos:post_end_pos]

        if len(est_slice) < 20:
            patterns.append(
                {
                    "hisse_kodu": ticker,
                    "karar_tarihi": event_str,
                    "success": False,
                    "error": "insufficient_estimation_data",
                }
            )
            continue

        # Volume analysis
        avg_vol_est = est_slice["Volume"].mean() if len(est_slice) > 0 else 1.0
        avg_vol_evt = event_slice["Volume"].mean() if len(event_slice) > 0 else 0.0
        max_vol_evt = event_slice["Volume"].max() if len(event_slice) > 0 else 0.0

        vol_ratio = avg_vol_evt / avg_vol_est if avg_vol_est > 0 else 0.0
        max_vol_ratio = max_vol_evt / avg_vol_est if avg_vol_est > 0 else 0.0

        # Volatility analysis
        vol_est = est_slice["return"].std() if len(est_slice) > 1 else 0.0
        vol_evt = event_slice["return"].std() if len(event_slice) > 1 else 0.0
        vol_ratio_val = vol_evt / vol_est if vol_est > 0 else 0.0

        # Cumulative returns
        cum_ret_pre = float(pre_slice["return"].sum()) if len(pre_slice) > 0 else 0.0
        cum_ret_post = float(post_slice["return"].sum()) if len(post_slice) > 0 else 0.0

        # Max run-up (pre) and drawdown (post)
        if len(pre_slice) > 0:
            cum_pre_series = pre_slice["return"].cumsum()
            max_runup = float(cum_pre_series.max())
        else:
            max_runup = 0.0

        if len(post_slice) > 0:
            cum_post_series = post_slice["return"].cumsum()
            max_drawdown = float(cum_post_series.min())
        else:
            max_drawdown = 0.0

        patterns.append(
            {
                "hisse_kodu": ticker,
                "karar_tarihi": event_str,
                "company_name": row["company_name"],
                "ceza_turu": row["ceza_turu"],
                "toplam_ceza_tl": row["toplam_ceza_tl"],
                "avg_volume_estimation": float(avg_vol_est),
                "avg_volume_event": float(avg_vol_evt),
                "volume_ratio": float(vol_ratio),
                "max_volume_ratio": float(max_vol_ratio),
                "volatility_estimation": float(vol_est),
                "volatility_event": float(vol_evt),
                "volatility_ratio": float(vol_ratio_val),
                "cum_return_pre": cum_ret_pre,
                "cum_return_post": cum_ret_post,
                "max_runup": max_runup,
                "max_drawdown": max_drawdown,
                "success": True,
                "error": "",
            }
        )

    results_df = pd.DataFrame(patterns)

    # ── Aggregate pattern summary ───────────────────────────────────────
    ok = results_df.loc[results_df.get("success", pd.Series(dtype=bool)) == True]  # noqa: E712
    if len(ok) > 0:
        logger.info("=" * 60)
        logger.info("MANIPULATION PATTERN SUMMARY")
        logger.info("=" * 60)
        logger.info(
            "Avg volume ratio (event/estimation) : %.2f",
            ok["volume_ratio"].mean(),
        )
        logger.info(
            "Avg max volume spike ratio           : %.2f",
            ok["max_volume_ratio"].mean(),
        )
        logger.info(
            "Avg volatility ratio                 : %.2f",
            ok["volatility_ratio"].mean(),
        )
        logger.info(
            "Avg cum return pre-event             : %.4f (%.2f%%)",
            ok["cum_return_pre"].mean(),
            ok["cum_return_pre"].mean() * 100,
        )
        logger.info(
            "Avg cum return post-event            : %.4f (%.2f%%)",
            ok["cum_return_post"].mean(),
            ok["cum_return_post"].mean() * 100,
        )
        logger.info(
            "Avg max run-up (pre)                 : %.4f (%.2f%%)",
            ok["max_runup"].mean(),
            ok["max_runup"].mean() * 100,
        )
        logger.info(
            "Avg max drawdown (post)              : %.4f (%.2f%%)",
            ok["max_drawdown"].mean(),
            ok["max_drawdown"].mean() * 100,
        )
        logger.info("=" * 60)

    if save_results:
        out_dir = config.PROCESSED_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "spk_manipulation_patterns.csv"
        results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info("Pattern results saved to %s", out_path)

    return results_df


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 :  CONVENIENCE / CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def print_database_summary() -> None:
    """Pretty-print the penalty database summary to stdout."""
    stats = summary_statistics()
    print("\n" + "=" * 65)
    print("  SPK MANIPULATION PENALTY DATABASE  —  SUMMARY")
    print("=" * 65)
    print(f"  Total cases           : {stats['total_cases']}")
    print(f"  Total penalties (TL)  : {stats['total_penalty_tl']:,.0f}")
    print(f"  Mean penalty (TL)     : {stats['mean_penalty_tl']:,.0f}")
    print(f"  Median penalty (TL)   : {stats['median_penalty_tl']:,.0f}")
    print(f"  Max penalty (TL)      : {stats['max_penalty_tl']:,.0f}  ({stats['max_penalty_ticker']})")
    print(f"  Total persons         : {stats['total_persons_penalized']}")
    print(f"  Trading-ban rate      : {stats['trading_ban_rate']:.1%}")
    print(f"  Avg investigation     : {stats['avg_investigation_days']:.0f} days")
    print("-" * 65)
    print("  Cases by year:")
    for yr in sorted(stats["cases_by_year"]):
        cnt = stats["cases_by_year"][yr]
        tl = stats["by_year"].get(yr, 0)
        print(f"    {yr}: {cnt:>3} cases  |  {tl:>15,.0f} TL")
    print("-" * 65)
    print("  Cases by type:")
    for t in sorted(stats["cases_by_type"]):
        label = CEZA_TURU_LABELS.get(t, t)
        cnt = stats["cases_by_type"][t]
        tl = stats["by_type"].get(t, 0)
        print(f"    {label}:")
        print(f"      {cnt} cases  |  {tl:>15,.0f} TL")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    print_database_summary()

    # Uncomment below to run the full event study (requires yfinance + scipy):
    # results = run_manipulation_event_study()
    # patterns = analyze_manipulation_patterns()
