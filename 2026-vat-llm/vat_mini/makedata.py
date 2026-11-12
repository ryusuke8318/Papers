"""合成 VAT 事例の確定的生成。`python makedata.py` で data_seed.json を作る。

注意: 規則文・事例はすべて合成 (synthetic paraphrase) であり、実際の
オーストリア VAT 法の条文・実事例ではない。各タイプに固有キーワードを
持たせ、語彙検索が正解規則に当たる設計 (hit 率は check で検証する)。
"""

from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).parent

# (規則 id, 引用ラベル, 本文)。本文は合成であり法文ではない。
RULES = [
    ("R-goods-1", "§ 3 Abs. 1 UStG (sinngemäß)",
     "Warenlieferung ohne Versand: Der Ort der Lieferung liegt dort, wo sich die Ware "
     "zum Zeitpunkt des Übergangs der Verfügungsmacht befindet. Entscheidend ist der Ort "
     "der physischen Übergabe an den Käufer."),
    ("R-goods-2", "§ 3 Abs. 1 UStG, ruhende Lieferung (sinngemäß)",
     "Bei der ruhenden Lieferung ohne Beförderung richtet sich der Lieferort nach dem "
     "Lagerort der Ware bei Vertragsabschluss. Der Käufer holt die Ware am Lager ab."),
    ("R-trans-1", "§ 3 Abs. 2 UStG, Versandlieferung (sinngemäß)",
     "Wird die Ware versendet oder befördert, liegt der Ort der Lieferung dort, wo sich "
     "die Ware bei Beginn der Versendung oder Beförderung befindet. Maßgeblich ist der "
     "Abgangsort des Transports."),
    ("R-trans-2", "§ 3 Abs. 2 UStG, Beförderung durch den Käufer (sinngemäß)",
     "Auch bei Abholung mit eigenem Beförderungsmittel (Selbstabholung mit Lkw) gilt der "
     "Beginn der Beförderung als Anknüpfungspunkt für den Lieferort."),
    ("R-b2b-1", "§ 3a Abs. 2 UStG, B2B-Grundleistung (sinngemäß)",
     "Dienstleistungen an Unternehmer (B2B) werden grundsätzlich dort erbracht, wo der "
     "leistungsempfangende Unternehmer sein Unternehmen betreibt. Es gilt das "
     "Empfängerortprinzip für Beratungs- und Werbeleistungen zwischen Unternehmen."),
    ("R-b2b-2", "§ 3a Abs. 2 UStG, grenzüberschreitende Beratung (sinngemäß)",
     "Beratungsleistungen eines inländischen Unternehmens an ein ausländisches Unternehmen "
     "sind am Sitz des ausländischen Auftraggebers steuerbar."),
    ("R-b2c-1", "§ 3a Abs. 1 UStG, B2C-Grundleistung (sinngemäß)",
     "Dienstleistungen an Nichtunternehmer (B2C) werden grundsätzlich dort erbracht, wo "
     "der leistende Unternehmer sein Unternehmen betreibt. Es gilt das Unternehmerortprinzip."),
    ("R-b2c-2", "§ 3a Abs. 1 UStG, Privatkunden (sinngemäß)",
     "Erbringt ein inländisches Unternehmen eine sonstige Leistung an eine Privatperson "
     "mit Wohnsitz im Ausland, bleibt der Leistungsort im Inland am Sitz des Leistenden."),
    ("R-chain-1", "Reihengeschäft, bewegte Lieferung (sinngemäß)",
     "Im Reihengeschäft mit nur einer grenzüberschreitenden Warenbewegung (Kettengeschäft) "
     "wird die bewegte Lieferung dem Transport zugeordnet. Nur diese Lieferung folgt dem "
     "Abgangsort; die ruhenden Lieferungen der Kette folgen dem Endpunkt der Beförderung."),
    ("R-chain-2", "Reihengeschäft, ruhende Lieferungen (sinngemäß)",
     "Die ruhenden Lieferungen innerhalb eines Kettengeschäfts gelten dort als ausgeführt, "
     "wo die Warenbewegung endet. Entscheidend ist der Bestimmungsort der bewegten Lieferung."),
    ("R-tri-1", "Dreiecksgeschäft, Vereinfachung (sinngemäß)",
     "Im innergemeinschaftlichen Dreiecksgeschäft mit drei Unternehmern aus drei Staaten "
     "wird der mittlere Unternehmer (Erwerber) von der Registrierung im Bestimmungsland "
     "befreit, wenn der Empfänger die Steuer schuldet. Die Kette endet im Land des Letztempfängers."),
    ("R-place-1", "Besteuerungsland, Grundsatz (sinngemäß)",
     "Die Umsatzsteuer steht dem Staat zu, in dem der Ort der Lieferung oder sonstigen "
     "Leistung liegt. Es gibt für jeden Umsatz nur ein einziges Besteuerungsland."),
]

FIRMEN = ["Alpen Maschinen GmbH", "Donau Tech AG", "Walhalla Logistik GmbH", "Brenner Handel OG",
          "Mozart Media GmbH", "Seefeld Consulting KG", "Inntal Bau GmbH", "Neusiedl Foods AG"]
WAREN = ["eine Industriemaschine", "eine Druckanlage", "einen Gabelstapler", "eine Serveranlage",
         "einen Traktor", "eine Textilmaschine", "einen Kran", "eine Solaranlage"]
LEISTUNGEN = ["eine Werbekampagne", "eine Steuerberatung", "eine IT-Beratung", "eine Marktstudie",
              "eine Designleistung", "eine Rechtsberatung", "eine Logistikberatung", "eine Schulung"]
STAEDTE = [("Wien", "Österreich"), ("Graz", "Österreich"), ("München", "Deutschland"),
           ("Mailand", "Italien"), ("Zagreb", "Kroatien"), ("Kopenhagen", "Dänemark"),
           ("Lyon", "Frankreich"), ("Salzburg", "Österreich")]

TEMPLATES = {
    # (template, gold_rule, gold_fn)
    "goods": [
        ("Die {firma} aus {stadt} verkauft {ware} an einen Käufer aus {kland}. "
         "Der Käufer übernimmt die Verfügungsmacht am Lager in {stadt}, ein Versand findet nicht statt. "
         "Wo liegt der Ort der ruhenden Warenlieferung?",
         "R-goods-1", lambda s: s[1]),
        ("Ein Käufer aus {kland} erwirbt {ware} und holt sie mit eigenem Fahrzeug am Lager in {stadt} ab. "
         "Der Übergang der Verfügungsmacht erfolgt in {stadt}. "
         "In welchem Land ist die Lieferung ohne Beförderung steuerbar?",
         "R-goods-2", lambda s: s[1]),
        ("Die {firma} veräußert {ware} aus ihrem Lager in {stadt} an einen Abnehmer aus {kland}. "
         "Transport oder Versand sind nicht vereinbart. Wo liegt der Lieferort?",
         "R-goods-1", lambda s: s[1]),
    ],
    "transport": [
        ("Die {firma} liefert {ware} von {stadt} an einen Kunden in {kland}. "
         "Der Versand beginnt in {stadt} mit dem Lkw. Wo liegt der Ort der Versandlieferung?",
         "R-trans-1", lambda s: s[1]),
        ("Ein Kunde aus {kland} bestellt {ware} bei der {firma} in {stadt}. "
         "Der Kunde übernimmt die Selbstabholung mit eigenem Lkw als Beförderungsmittel. "
         "Wo beginnt die Beförderung, und wo liegt daher der Lieferort?",
         "R-trans-2", lambda s: s[1]),
        ("Die {firma} in {stadt} versendet {ware} an einen Abnehmer in {kland}. "
         "Abgangsort des Transports ist {stadt}. In welchem Land ist diese Versandlieferung steuerbar?",
         "R-trans-1", lambda s: s[1]),
    ],
    "b2b": [
        ("Die {firma} aus {stadt} erbringt {leistung} für ein Unternehmen mit Sitz in {kland}. "
         "Wo liegt der Ort der B2B-Dienstleistung nach dem Empfängerortprinzip?",
         "R-b2b-1", lambda s: None),  # gold = Kundenland
        ("Ein österreichisches Beratungsunternehmen in {stadt} berät einen Auftraggeber aus {kland}. "
         "Es handelt sich um eine grenzüberschreitende Beratung zwischen Unternehmen. "
         "Wo ist die Leistung steuerbar?",
         "R-b2b-2", lambda s: None),
        ("Die {firma} erstellt {leistung} für einen Unternehmer in {kland}. "
         "Leistungsempfänger ist ein ausländisches Unternehmen. Wo liegt der Dienstleistungsort?",
         "R-b2b-1", lambda s: None),
    ],
    "b2c": [
        ("Die {firma} aus {stadt} erbringt {leistung} für eine Privatperson mit Wohnsitz in {kland}. "
         "Wo liegt der Ort der B2C-Leistung nach dem Unternehmerortprinzip?",
         "R-b2c-1", lambda s: s[1]),
        ("Eine in {stadt} ansässige Gesellschaft schult eine Privatperson aus {kland} vor Ort in {stadt}. "
         "Der Wohnsitz des Kunden liegt im Ausland. Wo ist die sonstige Leistung steuerbar?",
         "R-b2c-2", lambda s: s[1]),
        ("Die {firma} berät einen Nichtunternehmer aus {kland}. Die Beratung erfolgt am Sitz in {stadt}. "
         "Es handelt sich um eine B2C-Leistung nach dem Unternehmerortprinzip. Wo liegt der Leistungsort?",
         "R-b2c-1", lambda s: s[1]),
    ],
    "chain": [
        ("Kettengeschäft: Ein Hersteller aus {kland} verkauft {ware} an einen Händler in {stadt}, "
         "der sie sofort an einen Endkunden in {kstadt} weiterverkauft. Die einzige Warenbewegung "
         "führt von {stadt} nach {kstadt}. Wo liegt der Ort der bewegten Lieferung?",
         "R-chain-1", lambda s: s[1]),
        ("Reihengeschäft über drei Länder: Die Ware ruht zunächst in {stadt} und wird dann nach "
         "{kstadt} ({kland}) befördert. Wo gelten die ruhenden Lieferungen der Kette als ausgeführt?",
         "R-chain-2", lambda s: None),  # gold = Bestimmungsland
        ("Dreiecksgeschäft: Drei Unternehmer aus {stadt}, {kland} und {kstadt} handeln mit {ware}. "
         "Der mittlere Unternehmer ist im Bestimmungsland nicht registriert. "
         "In welchem Land endet die Kette steuerlich?",
         "R-tri-1", lambda s: None),
    ],
}


def build_pools(seed: int = 0, n_train_per_type: int = 300, n_test_per_type: int = 150) -> dict:
    """train pool (chain なし) + test pool (全タイプ) を確定的に作る。"""
    rng = random.Random(seed)
    train_pool, test_pool = [], []
    counters = {"train": 0, "test": 0}
    for typ, tmpls in TEMPLATES.items():
        for i in range(n_train_per_type if typ != "chain" else 0):
            train_pool.append(make_case(rng, counters, "train", typ, tmpls))
        for i in range(n_test_per_type):
            test_pool.append(make_case(rng, counters, "test", typ, tmpls))
    rng.shuffle(train_pool)
    rng.shuffle(test_pool)
    return {"train_pool": train_pool, "test_pool": test_pool}


def make_case(rng: random.Random, counters: dict, split: str, typ: str, tmpls: list) -> dict:
    tmpl, rule, gold_fn = tmpls[rng.randrange(len(tmpls))]
    firma = rng.choice(FIRMEN)
    ware = rng.choice(WAREN)
    leistung = rng.choice(LEISTUNGEN)
    (stadt, sland) = rng.choice(STAEDTE)
    (kstadt, kland) = rng.choice([c for c in STAEDTE if c[1] != sland])
    q = tmpl.format(firma=firma, stadt=stadt, kland=kland, kstadt=kstadt,
                    ware=ware, leistung=leistung)
    gold = gold_fn((stadt, sland))
    if gold is None:  # Kunden-/Bestimmungsland
        gold = kland
    cite = dict((r[0], r[1]) for r in RULES)[rule]
    answer = f"{gold}. {cite} zufolge liegt das einzige Besteuerungsland in {gold}."
    counters[split] += 1
    return {"id": f"{split}-{typ}-{counters[split]:04d}", "type": typ,
            "question": q, "gold": gold, "gold_rule": rule, "answer": answer}


def main() -> None:
    pools = build_pools()
    rules = [{"id": rid, "cite": cite, "text": text} for rid, cite, text in RULES]
    out = {"rules": rules, **pools,
           "note": "synthetic paraphrase, kein Rechts text. Nur zur Richtungsreplikation."}
    p = HERE / "data_seed.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {p}: rules={len(rules)} train={len(pools['train_pool'])} test={len(pools['test_pool'])}")


if __name__ == "__main__":
    main()
