"""
Enamine API Client for fetching compound prices.
Based on Enamine Store API v2 with batch processing.
Caches both valid and invalid compounds for faster subsequent queries.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors

# Configuration
BASE_URL = "https://www.enaminestore.com/api"
API_BATCH_SIZE = 2000
DEFAULT_MAX_PRICE_PER_GRAM = 40.0
DEFAULT_MAX_PRICE_PER_PACK = 250.0
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 2.0
API_REQUEST_DELAY = 0.5
API_TIMEOUT_SECONDS = 120

# Credentials — loaded from environment variables (set in .env file)
def _load_dotenv() -> None:
    """Load variables from .env file into os.environ (no external deps)."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

_load_dotenv()

_ENAMINE_EMAIL = os.environ.get("ENAMINE_EMAIL", "")
_ENAMINE_PASSWORD = os.environ.get("ENAMINE_PASSWORD", "")

# Cache directory
DEFAULT_CACHE_DIR = "mol_files/0. EnamineSDFs/price_cache"


class EnamineClient:
    """Client for Enamine Store API v2 with retry logic."""
    
    def __init__(self) -> None:
        self.session = requests.Session()
        self.tokens: dict[str, str | None] = {"access": None, "refresh": None}
        self._signed_in = False
    
    def _b64(self, s: str) -> str:
        """Encode string to base64."""
        return base64.b64encode(s.encode("utf-8")).decode("ascii")
    
    def sign_in(self) -> None:
        """
        Authenticate with Enamine API.
        
        Raises:
            RuntimeError: If authentication fails.
            ValueError: If credentials are not set.
        """
        if not _ENAMINE_EMAIL or not _ENAMINE_PASSWORD:
            raise ValueError(
                "Enamine credentials not found. "
                "Create a .env file in the project root with:\n"
                "  ENAMINE_EMAIL=your_email\n"
                "  ENAMINE_PASSWORD=your_password"
            )
        encoded_email = self._b64(_ENAMINE_EMAIL)
        encoded_password = self._b64(_ENAMINE_PASSWORD)
        
        try:
            resp = self.session.post(
                f"{BASE_URL}/v1/auth/as-customer/sign-in",
                json={"email": encoded_email, "password": encoded_password},
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            
            self.tokens["access"] = resp.cookies.get("ENAMINESTOREAUTH")
            self.tokens["refresh"] = data.get("refreshToken")
            
            if self.tokens["access"]:
                self.session.cookies.set("ENAMINESTOREAUTH", self.tokens["access"])
            
            self._signed_in = True
            print("[EnamineClient] Signed in successfully.")
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Sign-in failed: {e}")
    
    def refresh_tokens(self) -> None:
        """
        Refresh authentication tokens.
        
        Raises:
            RuntimeError: If refresh token is missing or refresh fails.
        """
        if not self.tokens["refresh"]:
            raise RuntimeError("No refresh token available.")
        
        try:
            resp = self.session.post(
                f"{BASE_URL}/v1/auth/refresh",
                json={"refreshToken": self.tokens["refresh"]},
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            
            self.tokens["access"] = resp.cookies.get("ENAMINESTOREAUTH")
            self.tokens["refresh"] = data.get("refreshToken")
            
            if self.tokens["access"]:
                self.session.cookies.set("ENAMINESTOREAUTH", self.tokens["access"])
        
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Token refresh failed: {e}")
    
    def fetch_batch(self, batch: list[str]) -> dict[str, Any] | None:
        """
        Fetch pricing data for a batch of compound IDs with retry logic.
        
        Parameters:
            batch: List of Enamine catalog IDs (tested up to 5000).
        
        Returns:
            Parsed JSON response or None if request fails.
        """
        if not self._signed_in:
            self.sign_in()
        
        url = f"{BASE_URL}/v2/catalog/search/by/codes"
        payload = {"compounds": batch, "currency": "EUR"}
        
        for attempt in range(API_RETRY_ATTEMPTS):
            try:
                resp = self.session.post(url, json=payload, timeout=API_TIMEOUT_SECONDS)
                
                if resp.status_code == 401:
                    self.refresh_tokens()
                    resp = self.session.post(url, json=payload, timeout=API_TIMEOUT_SECONDS)
                
                if resp.status_code == 400:
                    print(f"⚠️ Bad request for batch {batch[:3]}...")
                    return None
                
                resp.raise_for_status()
                return resp.json()
            
            except requests.exceptions.Timeout:
                if attempt < API_RETRY_ATTEMPTS - 1:
                    print(f"⚠️ Timeout on batch attempt {attempt + 1}/{API_RETRY_ATTEMPTS}, retrying...")
                    time.sleep(API_RETRY_DELAY)
                else:
                    print(f"⚠️ Timeout on batch after {API_RETRY_ATTEMPTS} attempts")
                    return None
            
            except requests.exceptions.RequestException as e:
                if attempt < API_RETRY_ATTEMPTS - 1:
                    print(f"⚠️ Request error (attempt {attempt + 1}/{API_RETRY_ATTEMPTS}): {e}")
                    time.sleep(API_RETRY_DELAY)
                else:
                    print(f"⚠️ Request failed after {API_RETRY_ATTEMPTS} attempts: {e}")
                    return None
        
        return None


def _extract_prices_from_batch_response(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """
    Extract prices from batch API response.
    
    Parameters:
        data: Raw JSON response from /v2/catalog/search/by/codes endpoint.
    
    Returns:
        Dict mapping compound codes to list of available price packs.
        Each pack contains: price, amount, measure, price_per_gram.
    """
    results: dict[str, list[dict[str, Any]]] = {}
    
    if not data or "results" not in data:
        return results
    
    for entry in data.get("results", []):
        product = entry.get("product", {})
        prices = entry.get("prices", {})
        
        code = product.get("code")
        if not code:
            continue
        
        all_prices = prices.get("g", {}).get("all", [])
        packs: list[dict[str, Any]] = []
        
        for price_entry in all_prices:
            price = price_entry.get("price")
            weight = price_entry.get("weight", {})
            measure = weight.get("measure")
            amount = weight.get("amount")
            
            if price is None or amount is None or measure is None:
                continue
            
            if measure == "mg":
                grams = amount / 1000.0
            elif measure == "g":
                grams = amount
            elif measure == "kg":
                grams = amount * 1000.0
            else:
                continue
            
            if grams > 0:
                packs.append({
                    "price": float(price),
                    "amount": float(amount),
                    "measure": measure,
                    "price_per_gram": float(price) / grams
                })
        
        results[code] = packs
    
    return results


def _find_best_pack(
    packs: list[dict[str, Any]],
    max_price_per_gram: float,
    max_pack_price: float
) -> dict[str, Any] | None:
    """
    Find best pack satisfying both price constraints.
    
    Logic:
    1. Filter packs with total price <= max_pack_price
    2. From remaining, select pack with lowest price_per_gram
    3. If lowest price_per_gram > max_price_per_gram, discard compound
    
    Parameters:
        packs: List of available price packs.
        max_price_per_gram: Maximum price per gram threshold.
        max_pack_price: Maximum total pack price threshold.
    
    Returns:
        Best pack (lowest price_per_gram within constraints) or None if no valid packs.
    """
    affordable_packs = [p for p in packs if p["price"] <= max_pack_price]
    
    if not affordable_packs:
        return None
    
    best_pack = min(affordable_packs, key=lambda x: x["price_per_gram"])
    
    if best_pack["price_per_gram"] > max_price_per_gram:
        return None
    
    return best_pack


def _load_cache(cache_file: str) -> dict[str, dict[str, Any]]:
    """
    Load cached prices from JSON file.
    
    Cache structure:
    {
        "valid": {compound_id: pack_data, ...},
        "invalid": {compound_id: True, ...}
    }
    
    Returns:
        Dict with "valid" and "invalid" keys, or empty dict if file doesn't exist.
    """
    if not os.path.exists(cache_file):
        return {"valid": {}, "invalid": {}}
    
    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)
            if data and isinstance(data, dict):
                if "valid" in data or "invalid" in data:
                    return data
                return {"valid": data, "invalid": {}}
            return {"valid": {}, "invalid": {}}
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️ Error reading cache {cache_file}: {e}")
        return {"valid": {}, "invalid": {}}


def _save_cache(cache_file: str, cache: dict[str, dict[str, Any]]) -> None:
    """
    Save prices to cache JSON file.
    
    Parameters:
        cache_file: Path to cache file.
        cache: Dict with "valid" and "invalid" keys.
    """
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
    except IOError as e:
        print(f"⚠️ Error saving cache {cache_file}: {e}")


def add_enamine_prices(
    df: pd.DataFrame,
    client: EnamineClient | None = None,
    id_col: str = "Catalog_ID",
    smiles_col: str = "SMILES",
    max_price_per_gram: float = DEFAULT_MAX_PRICE_PER_GRAM,
    max_pack_price: float = DEFAULT_MAX_PRICE_PER_PACK,
    batch_size: int = API_BATCH_SIZE,
    use_cache: bool = True,
    cache_file: str | None = None,
    force_refresh: bool = False,
    print_report: bool = True
) -> pd.DataFrame:
    """
    Add Enamine pricing to DataFrame with dual-tier caching.
    
    Queries API in batches. For each compound:
    1. Filter packs with total price <= max_pack_price
    2. From remaining, select pack with lowest price_per_gram
    3. If lowest price_per_gram > max_price_per_gram, discard compound
    
    Caches both valid prices and compounds with no valid pricing (invalid) 
    to avoid re-querying rejected compounds.
    
    Batch Size & Network Considerations:
    - Default: 2000 compounds (optimal stability + speed for most networks)
    - Options:
      * 2000 (default): works reliably everywhere
      * 3000-5000: faster on fast networks, may timeout on slow connections
      * 1000-1500: Use if experiencing timeout errors on slow networks
    - Larger batch = fewer API requests but bigger JSON payloads
    - Timeout: 120 seconds (increased from 60 for slow networks)
    - Retry logic: 3 attempts with 2 second delays between retries
    
    Cache structure:
    {
        "valid": {compound_id: {price, amount, measure, price_per_gram}, ...},
        "invalid": {compound_id: True, ...}
    }
    
    Parameters:
        df: DataFrame with Catalog_ID column
        client: EnamineClient instance (created if None)
        id_col: Column with Enamine catalog IDs (default: "Catalog_ID")
        smiles_col: Column with SMILES strings (default: "SMILES")
        max_price_per_gram: Max price per gram in EUR/g (default: 40)
        max_pack_price: Max total pack price in EUR (default: 250)
        batch_size: Number of compounds per API batch (default: 2000)
        use_cache: Use cached prices (default: True)
        cache_file: Cache file path (auto-generated if None)
        force_refresh: Ignore cache and re-query all (default: False)
        print_report: Print progress messages (default: True)
    
    Returns:
        DataFrame filtered to compounds with valid pricing, with columns:
        - PriceG: Price per gram (EUR/g)
        - PriceMol: Price per molecule (EUR, based on molecular weight)
    
    Raises:
        ValueError: If required columns are missing.
    """
    if id_col not in df.columns:
        raise ValueError(f"Missing column: {id_col}")
    if smiles_col not in df.columns:
        raise ValueError(f"Missing column: {smiles_col}")
    
    if client is None:
        client = EnamineClient()
        client.sign_in()
    
    if cache_file is None:
        sample_id = str(df[id_col].iloc[0]) if len(df) > 0 else "unknown"
        if sample_id.startswith("A"):
            cache_name = "aldehydes_prices.json"
        elif sample_id.startswith("C"):
            cache_name = "carboxylics_prices.json"
        else:
            cache_name = "compound_prices.json"
        cache_file = os.path.join(DEFAULT_CACHE_DIR, cache_name)
    
    cache = {}
    if use_cache and not force_refresh:
        cache = _load_cache(cache_file)
    else:
        cache = {"valid": {}, "invalid": {}}
    
    valid_cache = cache.get("valid", {})
    invalid_cache = cache.get("invalid", {})
    total_cached = len(valid_cache) + len(invalid_cache)
    
    if print_report and total_cached > 0:
        print(f"[Enamine Pricing] Loaded {len(valid_cache)} valid + {len(invalid_cache)} invalid cached")
    
    out_df = df.copy()
    out_df["PriceG"] = pd.NA
    out_df["PriceMol"] = pd.NA
    
    compound_ids = out_df[id_col].dropna().astype(str).tolist()
    
    if force_refresh or not use_cache:
        ids_to_query = compound_ids
        cached_count = 0
    else:
        ids_to_query = [
            cid for cid in compound_ids 
            if cid not in valid_cache and cid not in invalid_cache
        ]
        cached_count = len(compound_ids) - len(ids_to_query)
    
    if print_report:
        print(f"[Enamine Pricing] Processing {len(compound_ids)} compounds...")
        if cached_count > 0:
            print(f"[Enamine Pricing] Using cache for {cached_count} compounds")
        print(f"[Enamine Pricing] Querying {len(ids_to_query)} compounds via API")
        print(f"[Enamine Pricing] Filters: <= {max_price_per_gram} EUR/g, <= {max_pack_price} EUR/pack")
    
    new_valid_prices: dict[str, dict[str, Any]] = {}
    new_invalid_compounds: set[str] = set()
    
    if ids_to_query:
        n_batches = (len(ids_to_query) + batch_size - 1) // batch_size
        
        for i in range(0, len(ids_to_query), batch_size):
            batch = ids_to_query[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            if print_report:
                print(f"[Enamine Pricing] Batch {batch_num}/{n_batches}: {len(batch)} compounds")
            
            try:
                data = client.fetch_batch(batch)
                
                if data:
                    batch_results = _extract_prices_from_batch_response(data)
                    
                    for compound_id in batch:
                        packs = batch_results.get(compound_id, [])
                        
                        if packs:
                            best_pack = _find_best_pack(
                                packs, max_price_per_gram, max_pack_price
                            )
                            
                            if best_pack:
                                new_valid_prices[compound_id] = best_pack
                            else:
                                new_invalid_compounds.add(compound_id)
                        else:
                            new_invalid_compounds.add(compound_id)
                
                time.sleep(API_REQUEST_DELAY)
                
            except Exception as e:
                if print_report:
                    print(f"⚠️ Error in batch {batch_num}: {e}")
                for cid in batch:
                    new_invalid_compounds.add(cid)
    
    final_valid_cache = valid_cache.copy()
    final_valid_cache.update(new_valid_prices)
    final_invalid_cache = invalid_cache.copy()
    final_invalid_cache.update({cid: True for cid in new_invalid_compounds})

    mols = out_df[smiles_col].apply(Chem.MolFromSmiles)
    mws = mols.apply(lambda m: Descriptors.MolWt(m) if m else None)

    valid_ids = set(compound_ids) & set(final_valid_cache.keys())
    price_map = {
        cid: (final_valid_cache[cid]["price_per_gram"], 
              final_valid_cache[cid]["price_per_gram"] * mws.iloc[i])
        for i, cid in enumerate(compound_ids)
        if cid in valid_ids
        and final_valid_cache[cid]["price_per_gram"] <= max_price_per_gram
        and final_valid_cache[cid]["price"] <= max_pack_price
    }

    out_df["PriceG"] = out_df[id_col].map(lambda x: price_map.get(x, (None, None))[0])
    out_df["PriceMol"] = out_df[id_col].map(lambda x: price_map.get(x, (None, None))[1])
    
    if use_cache and (new_valid_prices or new_invalid_compounds):
        cache_to_save = {
            "valid": final_valid_cache,
            "invalid": final_invalid_cache
        }
        _save_cache(cache_file, cache_to_save)
        if print_report:
            print(f"[Enamine Pricing] Saved {len(final_valid_cache)} valid + {len(final_invalid_cache)} invalid to cache")
    
    initial_count = len(out_df)
    valid_mask = out_df["PriceG"].notna() & (out_df["PriceG"] > 0)
    out_df = out_df[valid_mask].reset_index(drop=True)
    removed_count = initial_count - len(out_df)
    
    if print_report:
        print(f"[Enamine Pricing] Completed: {len(out_df)}/{initial_count} with valid pricing")
        if removed_count > 0:
            zero_count = (out_df["PriceG"] == 0).sum() if len(out_df) > 0 else 0
            if zero_count > 0:
                print(f"⚠️  Removed {zero_count} compounds with zero/negative prices")
            else:
                print(f"⚠️  Removed {removed_count} compounds (no valid pricing)")
    
    return out_df
