
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from data import klanten, producten, aankoopgeschiedenis

# Functie: Vind promoties/combo-deals gebaseerd op aankooppatronen
# Voor eenvoud: zoek producten die vaak samen gekocht worden

def find_combo_deals(user_id):
    # Debug-print pas na definitie van voorkeuren
    desired = 3

    user_orders = aankoopgeschiedenis.get(user_id, [])
    user_products = [item for order in user_orders for item in order]
    unique_user_products = set(user_products)

    product_by_id: Dict[str, dict] = {p['id']: p for p in producten}
    user_categories = {
        product_by_id[p]['categorie']
        for p in unique_user_products
        if p in product_by_id and product_by_id[p].get('categorie')
    }

    klant = next((k for k in klanten if k['id'] == user_id), None)
    voorkeuren = set(klant.get('voorkeuren', [])) if klant else set()

    print(f"\n--- COMBO DEBUG ---")
    print(f"Klant-ID: {user_id}")
    print(f"Voorkeuren: {voorkeuren}")
    print(f"Unieke producten klant: {unique_user_products}")
    desired = 3

    user_orders = aankoopgeschiedenis.get(user_id, [])
    user_products = [item for order in user_orders for item in order]
    unique_user_products = set(user_products)

    product_by_id: Dict[str, dict] = {p['id']: p for p in producten}
    user_categories = {
        product_by_id[p]['categorie']
        for p in unique_user_products
        if p in product_by_id and product_by_id[p].get('categorie')
    }

    klant = next((k for k in klanten if k['id'] == user_id), None)
    voorkeuren = set(klant.get('voorkeuren', [])) if klant else set()

    print(f"\n--- COMBO DEBUG ---")
    print(f"Klant-ID: {user_id}")
    print(f"Voorkeuren: {voorkeuren}")
    print(f"Unieke producten klant: {unique_user_products}")

    pair_counts: Counter[Tuple[str, str]] = Counter()
    product_counts: Counter[str] = Counter()
    pairs_by_source: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

    for orders in aankoopgeschiedenis.values():
        for order in orders:
            product_counts.update(order)
            for i in range(len(order)):
                for j in range(i + 1, len(order)):
                    a, b = order[i], order[j]
                    if a == b:
                        continue
                    pair = tuple(sorted((a, b)))
                    pair_counts[pair] += 1

    for (p1, p2), count in pair_counts.items():
        pairs_by_source[p1].append((p2, count))
        pairs_by_source[p2].append((p1, count))

    deals = []
    added_targets: set[str] = set()

    def score_combo(source_id: str, target_id: str, base_score: int) -> float:
        target_prod = product_by_id.get(target_id)
        if not target_prod:
            return 0.0
        kenmerken = set(target_prod.get('kenmerken', []))
        preference_bonus = 2 * len(voorkeuren.intersection(kenmerken))
        diversity_bonus = 0.5 if target_prod.get('categorie') not in user_categories else 0.0
        return float(base_score + preference_bonus + diversity_bonus)

    for source_id in unique_user_products:
        for target_id, count in sorted(pairs_by_source.get(source_id, []), key=lambda x: x[1], reverse=True):
            if target_id in unique_user_products or target_id in added_targets:
                continue
            combo_score = score_combo(source_id, target_id, count)
            if combo_score <= 0:
                continue
            deals.append((source_id, target_id, combo_score))
            added_targets.add(target_id)
            if len(deals) >= desired:
                break
        if len(deals) >= desired:
            break


    # Fallback: personaliseer met klantvoorkeuren en categorieën
    if len(deals) < desired:
        # Zoek producten die passen bij voorkeuren/categorieën van de klant
        global_pairs = sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)
        for (p1, p2), count in global_pairs:
            for source_id, target_id in ((p1, p2), (p2, p1)):
                if target_id in added_targets:
                    continue
                target_prod = product_by_id.get(target_id)
                if not target_prod:
                    continue
                # Extra bonus als het product in voorkeuren/categorieën valt
                bonus = 0
                if target_prod.get('subcategorie') in voorkeuren or target_prod.get('categorie') in voorkeuren:
                    bonus += 2
                if target_prod.get('categorie') in user_categories:
                    bonus += 1
                combo_score = score_combo(source_id, target_id, count + bonus)
                if combo_score <= 0:
                    continue
                deals.append((source_id, target_id, combo_score))
                added_targets.add(target_id)
                if len(deals) >= desired:
                    break
            if len(deals) >= desired:
                break

    # Laatste fallback: varieer per klant, randomiseer en geef voorkeuren extra gewicht
    import random
    if len(deals) < desired:
        # Kies alleen producten die de klant nog NIET heeft gekocht
        onbekochte_producten = [p['id'] for p in producten if p['id'] not in unique_user_products]
        # Geef voorkeuren extra gewicht
        voorkeur_producten = [p['id'] for p in producten if p.get('subcategorie') in voorkeuren or p.get('categorie') in voorkeuren]
        # Combineer en randomiseer
        kandidaten = list(set(onbekochte_producten + voorkeur_producten))
        random.shuffle(kandidaten)
        for anchor_id in list(unique_user_products):
            for target_id in kandidaten:
                if target_id == anchor_id or target_id in added_targets:
                    continue
                target_prod = product_by_id.get(target_id)
                if not target_prod:
                    continue
                # Score: extra bonus voor voorkeuren, random factor voor variatie
                bonus = 0
                if target_prod.get('subcategorie') in voorkeuren or target_prod.get('categorie') in voorkeuren:
                    bonus += 5
                bonus += random.uniform(0, 2)
                combo_score = score_combo(anchor_id, target_id, 1 + bonus)
                if combo_score <= 0:
                    continue
                deals.append((anchor_id, target_id, combo_score))
                added_targets.add(target_id)
                if len(deals) >= desired:
                    break
            if len(deals) >= desired:
                break

    detailed_deals = []
    for src_id, tgt_id, score in deals[:desired]:
        src_prod = product_by_id.get(src_id)
        tgt_prod = product_by_id.get(tgt_id)
        if src_prod and tgt_prod:
            detailed_deals.append((src_prod, tgt_prod, score))

    print(f"Combo deals voor klant {user_id}:")
    for src_prod, tgt_prod, score in detailed_deals:
        print(f"  {src_prod['naam']} + {tgt_prod['naam']} (score: {score})")
    print(f"--- EINDE COMBO DEBUG ---\n")
    return detailed_deals

# Functie: Haal productdetails
def get_product_by_id(pid):
    for p in producten:
        if p['id'] == pid:
            return p
    return None
