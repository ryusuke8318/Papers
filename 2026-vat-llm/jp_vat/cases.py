"""内外判定(国内取引か国外取引か)の事例を、条文・通達の規定から決定的に生成する。

各事例は family(規定の種類)ごとのテンプレートから作り、正解(国内 / 国外)と根拠(sources/corpus.json の unit id)は規定から機械的に決まる。
ただし条文の読みを私が符号化したものなので、税務の専門家によるレビューは受けていない(README に明記)。

分割
  train           seen の family(+ seen の例外)。fact 節の言い回しは v0, v1、地名は train 用の集合
  test_seen       seen の family。fact 節の言い回し v2(学習に出ない)と、学習に出ない地名
  test_exc_seen   学習に出る例外(通達 5-7-1, 5-7-10, 5-7-12)の新しい事例
  test_exc_unseen 学習に出ない例外(通達 5-7-11 船荷証券、令6②六 国内外にわたる役務)
  real            国税庁の質疑応答事例 9 件(sources/qa_raw.json の照会要旨。結論は人手)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).parent
DOM_TRAIN = ["東京都", "大阪市", "名古屋市", "福岡市", "札幌市", "横浜市", "神戸市", "仙台市"]
DOM_TEST = ["広島市", "京都市", "さいたま市"]
FOR_TRAIN = ["ベトナムのホーチミン", "米国のロサンゼルス", "ドイツのハンブルク", "シンガポール", "韓国の釜山", "タイのバンコク",
             "英国のロンドン", "中国の上海"]
FOR_TEST = ["オーストラリアのシドニー", "ブラジルのサンパウロ", "インドのムンバイ", "フランスのパリ"]
FLAG = ["パナマ", "リベリア", "マーシャル諸島", "香港"]  # 船舶の外国の登録国
AC_FOR = ["米国", "アイルランド", "シンガポール"]
A = ["青山", "川口", "東雲", "北原", "三星", "丸山", "岡部", "白石", "高砂", "大和田", "桜井", "長峰"]
B = ["商事", "工業", "製作所", "物産", "運輸", "設計", "システムズ", "ソフト", "電機", "海運"]
FN = ["Alder Trading", "Brightway Logistics", "Cobalt Systems", "Dunmore Marine", "Elmwood Design", "Fairhaven Software",
      "Granite Works", "Harbor Light"]


class G:
    """1 回の生成で使う乱数と地名の集合。"""

    def __init__(self, rng, held_out: bool):
        self.r, self.ho = rng, held_out
        self.dom = DOM_TEST if held_out else DOM_TRAIN
        self.foreign = FOR_TEST if held_out else FOR_TRAIN

    def d(self):
        return self.r.choice(self.dom)

    def f(self):
        return self.r.choice(self.foreign)

    def jp(self):
        return self.r.choice(["株式会社", "有限会社"]) + self.r.choice(A) + self.r.choice(B)

    def fc(self):
        return f"{self.f().split('の')[0]}法人の{self.r.choice(FN)}社"

    def decoy(self):
        c = self.r.choice([
            f"契約書は{self.r.choice(self.dom + self.foreign)}で締結された。",
            f"代金は{self.r.choice(['円','米ドル','ユーロ'])}建てで、{self.r.choice(self.dom + self.foreign)}の銀行口座に振り込まれる。",
            f"取引金額は{self.r.choice([120, 480, 950, 3200])}万円である。",
            f"打合せは主にオンラインで行われた。",
            "",
        ])
        return c


def case(fam, g, label, gold, reason, text, typ="main"):
    return {"family": fam, "type": typ, "text": text, "label": label, "gold": gold, "reason": reason}


YN = {True: "国内", False: "国外"}


def dom_word(b):
    return "国内取引" if b else "国外取引"


# ---------------------------------------------------------------- 主な family(fact 節の言い回しは v=0,1,2)
def f_goods(g, v):
    inl = g.r.random() < 0.5
    loc = g.d() if inl else g.f()
    a, b = g.jp(), g.fc() if g.r.random() < .5 else g.jp()
    kind = g.r.choice(["譲渡", "貸付け"])
    item = g.r.choice(["工作機械", "業務用冷蔵設備", "建設資材", "印刷機", "中古の測定機器"])
    when = "引渡しの時" if kind == "譲渡" else "貸付けの時"
    t = [f"{a}は、{loc}の倉庫に保管している{item}を{b}に{kind.replace('貸付け','貸し付けた').replace('譲渡','売却した')}。この{item}は{when}に{loc}の倉庫にあった。",
         f"{a}は{b}との間で{item}の{kind}を行った。{item}は{when}、{loc}にある{a}の倉庫に置かれていた。",
         f"{b}は、{a}から{item}の{kind}を受けた。対象の{item}は{when}に{loc}の保管場所に所在していた。"][v]
    r = f"資産の{kind}は、{when}に資産が所在していた場所で判定する。{loc}に所在していたので{dom_word(inl)}。"
    return case("goods", g, inl, ["法-4-3-1"], r, t + g.decoy())


def f_ship(g, v):
    sub = g.r.choice("abcd")
    co, fo = g.jp(), g.fc()
    flag = g.r.choice(FLAG)
    ship = g.r.choice(["貨物船", "タンカー", "コンテナ船"])
    if sub == "a":
        lab, t = True, [f"{co}は、日本で登録を受けている{ship}を{fo}に譲渡した。", f"日本の登録船舶である{ship}を、{co}が{fo}へ売却した。",
                        f"{fo}は、{co}から、日本の船舶登録を受けた{ship}の譲渡を受けた。"][v]
        r = "譲渡する船舶は日本で登録を受けた日本船舶で、登録をした機関の所在地は国内。居住者が行う日本船舶の譲渡なので国内取引。"
    elif sub == "b":
        lab, t = False, [f"{co}は、{flag}で登録を受けている{ship}を{fo}に譲渡した。", f"{flag}籍(登録)の{ship}を{co}が{fo}へ売却した。",
                         f"{fo}は、{co}から、{flag}の登録を受けた{ship}の譲渡を受けた。"][v]
        r = f"船舶は登録をした機関の所在地で判定する。{flag}で登録されているので国外取引。"
    elif sub == "c":
        lab, t = False, [f"{fo}は、日本で登録を受けている{ship}を{co}に譲渡した。", f"日本船舶である{ship}を、{fo}が{co}へ売却した。",
                         f"{co}は、{fo}から、日本の登録船舶である{ship}の譲渡を受けた。"][v]
        r = "非居住者が行う日本船舶の譲渡は、譲渡を行う者の住所地(本店の所在地)で判定する。譲渡者は国外の法人なので国外取引。"
    else:
        lab, t = True, [f"{co}は、{flag}で登録を受けている{ship}を{fo}に貸し付けた。", f"{flag}籍の{ship}を{co}が{fo}に貸付けした。",
                        f"{fo}は、{co}から、{flag}の登録を受けた{ship}の貸付けを受けた。"][v]
        r = "居住者が行う日本船舶以外の船舶の貸付けは、貸付けを行う者の住所地(本店の所在地)で判定する。貸付者は国内の法人なので国内取引。"
    return case("ship", g, lab, ["令-6-1-1"], r, t + g.decoy())


def f_aircraft(g, v):
    inl = g.r.random() < .5
    reg = "日本" if inl else g.r.choice(AC_FOR)
    co, fo = g.jp(), g.fc()
    t = [f"{co}は、{reg}で登録を受けた航空機を{fo}に譲渡した。", f"{reg}の航空機登録を受けている機体を、{co}が{fo}へ売却した。",
         f"{fo}は、{co}から、{reg}で登録されている航空機の譲渡を受けた。"][v]
    r = f"航空機は登録をした機関の所在地で判定する。{reg}で登録されているので{dom_word(inl)}。"
    return case("aircraft", g, inl, ["令-6-1-3"], r, t + g.decoy())


def f_patent(g, v):
    sub = g.r.choice("abcd")
    seller_dom = g.r.random() < .5
    seller = g.jp() if seller_dom else g.fc()
    kind = g.r.choice(["特許権", "商標権", "意匠権"])
    other = g.r.choice(["米国", "ドイツ", "韓国", "中国"])
    buyer = g.jp() if g.r.random() < .5 else g.fc()
    if sub == "a":
        lab, t = True, [f"{seller}は、日本の特許庁等にのみ登録している{kind}を{buyer}に譲渡した。", f"{buyer}は、{seller}から、日本でのみ登録された{kind}の譲渡を受けた。",
                        f"{seller}が{buyer}に譲渡した{kind}は、日本にだけ登録されている。"][v]
        r = f"{kind}等は、権利の登録をした機関の所在地で判定する。日本のみで登録されているので国内取引。"
    elif sub == "b":
        lab, t = False, [f"{seller}は、{other}にのみ登録している{kind}を{buyer}に譲渡した。", f"{buyer}は、{seller}から、{other}でのみ登録された{kind}の譲渡を受けた。",
                         f"{seller}が{buyer}に譲渡した{kind}は、{other}にだけ登録されている。"][v]
        r = f"{kind}等は、権利の登録をした機関の所在地で判定する。{other}のみで登録されているので国外取引。"
    else:
        lab, t = seller_dom, [f"{seller}は、日本と{other}の双方に登録している{kind}を{buyer}に譲渡した。",
                              f"{buyer}は、{seller}から、日本と{other}の両方で登録された{kind}の譲渡を受けた。",
                              f"{seller}が{buyer}に譲渡した{kind}は、日本と{other}に重ねて登録されている。"][v]
        r = f"同じ権利を二以上の国で登録している場合は、権利の譲渡を行う者の住所地で判定する。譲渡者の所在は{'国内' if seller_dom else '国外'}なので{dom_word(seller_dom)}。"
    return case("patent", g, lab, ["令-6-1-5"], r, t + g.decoy())


def f_copyright(g, v):
    seller_dom = g.r.random() < .5
    seller = g.jp() if seller_dom else g.fc()
    buyer = g.fc() if seller_dom else g.jp()
    what = g.r.choice(["ソフトウェアの著作権", "楽曲の著作権", "出版権", "製造ノウハウ(特別の技術による生産方式)"])
    kind = g.r.choice(["譲渡", "貸付け"])
    t = [f"{seller}は、{what}を{buyer}に{kind}した。", f"{buyer}は、{seller}から{what}の{kind}を受けた。",
         f"{seller}と{buyer}の間で、{what}の{kind}が行われた。"][v]
    r = f"著作権等・特別の技術による生産方式は、{kind}を行う者の住所地で判定する。{kind}を行う{seller}は{'国内' if seller_dom else '国外'}の事業者なので{dom_word(seller_dom)}。"
    return case("copyright", g, seller_dom, ["令-6-1-7"], r, t + g.decoy())


def f_goodwill(g, v):
    dom = g.r.random() < .5
    seller = g.jp() if dom else g.fc()
    buyer = g.fc() if dom else g.jp()
    t = [f"{seller}は、営業権(許可に基づく営業の権利)を{buyer}に譲渡した。", f"{buyer}は、{seller}から営業権の譲渡を受けた。",
         f"{seller}が営業権を{buyer}へ売却した。この営業権に係る事業を行っているのは{seller}である。"][v]
    r = f"営業権は、営業権に係る事業を行う者の住所地で判定する。事業を行う{seller}は{'国内' if dom else '国外'}の事業者なので{dom_word(dom)}。"
    return case("goodwill", g, dom, ["令-6-1-8"], r, t + g.decoy())


def f_service(g, v):
    inl = g.r.random() < .5
    loc = g.d() if inl else g.f()
    a, c = g.jp(), g.fc() if g.r.random() < .5 else g.jp()
    kind = g.r.choice(["経営コンサルティング", "従業員向けの研修", "映像の撮影", "通訳", "機械の据付作業"])
    t = [f"{a}は、{c}の依頼により、{loc}で{kind}を行った。役務の提供を行った場所は{loc}である。",
         f"{c}は、{a}に{kind}を依頼した。{a}は{loc}の現地で役務を提供した。",
         f"{a}は{c}に対して{kind}の役務を提供した。契約上も実際にも、提供場所は{loc}で特定されている。"][v]
    r = f"役務の提供は、役務の提供が行われた場所で判定する。提供場所が{loc}と特定できるので{dom_word(inl)}。"
    return case("service", g, inl, ["法-4-3-2", "通達-5-7-15"], r, t + g.decoy())


def f_transport(g, v):
    kind = g.r.choice(["貨物", "旅客"])
    pat = g.r.choice(["dom_to_for", "for_to_dom", "for_to_for"])
    d, f1, f2 = g.d(), g.f(), g.f()
    while f2 == f1:
        f2 = g.f()
    ends = {"dom_to_for": (d, f1), "for_to_dom": (f1, d), "for_to_for": (f1, f2)}[pat]
    inl = pat != "for_to_for"
    co = g.jp() if g.r.random() < .5 else g.fc()
    t = [f"{co}は、{ends[0]}を出発地、{ends[1]}を到着地とする{kind}の国際輸送を行った。",
         f"{co}が請け負った{kind}の輸送は、{ends[0]}から{ends[1]}にわたって行われた。",
         f"{co}は、{ends[0]}で{kind}を受け取り、{ends[1]}まで運んだ。"][v]
    r = f"国内と国外にわたる{kind}の輸送は、出発地又は到着地が国内であれば国内取引。" + (f"{d}が{'出発地' if pat=='dom_to_for' else '到着地'}なので国内取引。" if inl else "出発地も到着地も国外なので国外取引。")
    return case("transport", g, inl, ["令-6-2-1", "通達-5-7-13"], r, t + g.decoy())


def f_comm(g, v):
    pat = g.r.choice(["dom_to_for", "for_to_dom", "for_to_for"])
    d, f1, f2 = g.d(), g.f(), g.f()
    while f2 == f1:
        f2 = g.f()
    ends = {"dom_to_for": (d, f1), "for_to_dom": (f1, d), "for_to_for": (f1, f2)}[pat]
    inl = pat != "for_to_for"
    co = g.jp() if g.r.random() < .5 else g.fc()
    t = [f"{co}は、発信地を{ends[0]}、受信地を{ends[1]}とする国際通信の役務を提供した。",
         f"{co}が提供した通信は、{ends[0]}から{ends[1]}へのものである。",
         f"{co}は、{ends[0]}で発信され{ends[1]}で受信される通信サービスを提供した。"][v]
    r = "国内と国外にわたる通信は、発信地又は受信地が国内であれば国内取引。" + (f"{d}が{'発信地' if pat=='dom_to_for' else '受信地'}なので国内取引。" if inl else "発信地も受信地も国外なので国外取引。")
    return case("comm", g, inl, ["令-6-2-2", "通達-5-7-13"], r, t + g.decoy())


def f_insurance(g, v):
    inl = g.r.random() < .5
    loc = g.d() if inl else g.f()
    ins, cl = g.fc() if not inl else g.jp(), g.jp()
    t = [f"保険会社{ins}は、{cl}との保険契約を、{loc}にある自社の事務所で締結した。", f"{cl}は、{ins}と保険契約を結んだ。契約の締結に係る{ins}の事務所は{loc}にある。",
         f"{ins}は保険に係る事業を営んでいる。{cl}との契約締結に係る事務所は{loc}である。"][v]
    r = f"保険は、保険事業を営む者の契約締結に係る事務所等の所在地で判定する。{loc}にあるので{dom_word(inl)}。"
    return case("insurance", g, inl, ["令-6-2-4"], r, t + g.decoy())


def f_expert(g, v):
    inl = g.r.random() < .5
    loc = g.d() if inl else g.f()
    other = g.f() if inl else g.d()
    a, c = g.jp(), g.jp()
    obj = g.r.choice(["工場", "発電所", "ビル", "港湾施設"])
    work = g.r.choice(["設計の助言", "施工の監督", "建設計画の立案"])
    t = [f"{a}は{c}から{obj}の建設に関する{work}(専門的な科学技術の知識を要する)を請け負った。建設に必要な資材の大部分は{loc}で調達される。{obj}は{other}に建設される。",
         f"{c}は{a}に{obj}の{work}を依頼した。専門的な科学技術に関する知識を必要とする役務で、資材の大部分の調達先は{loc}、建設地は{other}である。",
         f"{a}の{work}は、{other}で建設する{obj}に関するものである。建設資材の大部分は{loc}で調達する。"][v]
    r = f"専門的な科学技術に関する知識を必要とする建設等の役務は、必要な資材の大部分が調達される場所で判定する。{loc}で調達されるので{dom_word(inl)}。"
    return case("expert", g, inl, ["令-6-2-5"], r, t + g.decoy())


def f_telecom(g, v):
    sub = g.r.choice("abcd")
    prov = g.fc() if g.r.random() < .5 else g.jp()
    svc = g.r.choice(["アプリの配信", "クラウドサービス", "電子書籍の配信", "ソフトウェアの利用提供"])
    if sub == "a":
        lab, t = True, [f"{prov}は、日本に住所を有する個人に{svc}を行った。この個人は提供の間、海外に滞在していた。",
                        f"{prov}の{svc}を、日本に住所のある個人が海外出張中に利用した。",
                        f"日本に居住する個人が、旅行先の{g.f()}で{prov}から{svc}を受けた。"][v]
        r = "電気通信利用役務の提供は、提供を受ける者の住所等が国内にあるかで判定する。受ける者は日本に住所があり、滞在先が国外でも国内取引。"
    elif sub == "b":
        lab, t = False, [f"{prov}は、{g.f()}に住所を有する個人に{svc}を行った。", f"{g.f()}に居住する個人が、{prov}から{svc}を受けた。",
                         f"{prov}の{svc}を、国外に住所のある個人が利用した。"][v]
        r = "電気通信利用役務の提供は、提供を受ける者の住所等が国内にあるかで判定する。受ける者の住所は国外なので国外取引。"
    elif sub == "c":
        lab, t = True, [f"{prov}は、内国法人である{g.jp()}の{g.f()}にある事務所に対して{svc}を行った。",
                        f"内国法人の{g.f()}支店が、{prov}から{svc}を受けた。",
                        f"{prov}の{svc}を、本店が日本にある法人の国外事務所が利用した。"][v]
        r = "電気通信利用役務の提供は、提供を受ける者の本店又は主たる事務所の所在地で判定する。内国法人は本店が国内にあり、国外の事務所に対する提供でも国内取引。"
    else:
        lab, t = False, [f"{prov}は、{g.f()}に本店を有する法人に{svc}を行った。", f"{g.f()}に本店のある法人が、{prov}から{svc}を受けた。",
                         f"{prov}の{svc}を、本店が国外にある法人が利用した。"][v]
        r = "電気通信利用役務の提供は、提供を受ける者の本店又は主たる事務所の所在地で判定する。本店は国外なので国外取引。"
    return case("telecom", g, lab, ["法-4-3-3", "通達-5-7-15の2"], r, t + g.decoy())


# ---------------------------------------------------------------- 例外
def x_overseas_resale(g, v):  # 通達 5-7-1
    co, c2 = g.jp(), g.jp()
    f = g.f()
    item = g.r.choice(["衣料品", "電子部品", "家具", "食料品"])
    brought = g.r.random() < .4
    if not brought:
        t = [f"{co}は、{f}で購入した{item}を、日本に搬入することなく{f}の{c2}に譲渡した。経理処理は国内の仕入れとして行った。",
             f"{co}は{f}の倉庫にある{item}を買い付け、その{item}を国内へ持ち込まないまま{c2}へ転売した。",
             f"{co}が国外で仕入れた{item}は、{co}の経理上は国内取引として処理されたが、現物は国内に搬入されず{c2}に引き渡された。"][v]
        r = "国外で購入した資産を国内に搬入せず他へ譲渡した場合、経理処理のいかんを問わず国内において行った資産の譲渡等に該当しない。国外取引。"
    else:
        d = g.d()
        t = [f"{co}は、{f}で購入した{item}を日本に搬入し、{d}の倉庫に保管した後、{c2}に譲渡した。譲渡の時、{item}は{d}の倉庫にあった。",
             f"{co}は{f}から{item}を輸入して{d}に置き、その{item}を{d}で{c2}へ転売した。",
             f"{co}が国外で仕入れた{item}は、国内に搬入されて{d}で{c2}に引き渡された。"][v]
        r = f"国外で購入した資産でも、国内に搬入された後に譲渡する場合は、譲渡の時に資産が所在していた場所で判定する。{d}に所在していたので国内取引。"
    return case("x_resale", g, brought, ["通達-5-7-1", "法-4-3-1"], r, t + g.decoy(), "exc_seen")


def x_domestic_dealer(g, v):  # 通達 5-7-10
    co, c2 = g.jp(), g.jp()
    f = g.f()
    d = g.d()
    item = g.r.choice(["工作機械", "原材料", "重機"])
    inl = g.r.random() < .4
    place = d if inl else f
    t = [f"国内の事業者である{co}は、国内の事業者{c2}に対し、{place}に所在する{item}を譲渡した。対価は受領している。",
         f"{co}(国内の事業者)が、同じく国内の事業者である{c2}へ、{place}にある{item}を売却した。",
         f"{c2}は国内の事業者で、{co}(国内の事業者)から{place}所在の{item}を買い受けた。"][v]
    if inl:
        r = f"資産の譲渡は譲渡の時に資産が所在していた場所で判定する。{d}に所在していたので国内取引。"
    else:
        r = "国内の事業者が国内の他の事業者に対し、国外に所在する資産の譲渡をした場合も、国外において行われたこととなり課税の対象とならない。国外取引。"
    return case("x_domdealer", g, inl, ["通達-5-7-10", "法-4-3-1"], r, t + g.decoy(), "exc_seen")


def x_moved_lease(g, v):  # 通達 5-7-12
    to_dom = g.r.random() < .5
    start = g.f() if to_dom else g.d()
    end = g.d() if to_dom else g.f()
    co, c2 = g.jp(), g.jp()
    item = g.r.choice(["建設機械", "発電機", "測定装置"])
    t = [f"{co}は{c2}に{item}を貸し付けている。賃貸借契約では使用場所が{start}と特定されていたが、当事者の合意で使用場所を{end}に変更した。",
         f"{c2}は{co}から{item}を借りている。契約で使用場所は{start}と定めていたが、双方の合意により{end}へ使用場所を変更した。",
         f"{co}と{c2}の{item}の賃貸借契約は、当初の使用場所が{start}だった。合意に基づき使用場所は{end}に変わった。"][v]
    r = f"貸付けに係る資産の使用場所が契約で特定されている場合、合意で使用場所を変更したときは、変更後の使用場所が国内かどうかで改めて判定する。{end}に変わったので{dom_word(to_dom)}。"
    return case("x_movelease", g, to_dom, ["通達-5-7-12", "法-4-3-1"], r, t + g.decoy(), "exc_seen")


def u_bol(g, v):  # 通達 5-7-11(学習に出ない)
    dom_port = g.r.random() < .5
    port = g.d() if dom_port else g.f()
    co, c2 = g.jp(), g.jp()
    goods = g.r.choice(["機械", "穀物", "鋼材"])
    if dom_port:
        t = [f"{co}は、{goods}の貨物を表彰する船荷証券を{c2}に譲渡した。船荷証券に表示された荷揚地は{port}で、貨物は現在洋上にある。船荷証券の写しは保存している。",
             f"{c2}は、{co}から{goods}の船荷証券の譲渡を受けた。荷揚地(PORT OF DISCHARGE)は{port}で、譲渡の時、貨物は海上を輸送中だった。写しの保存がある。"][v % 2]
        r = "船荷証券の譲渡は貨物の譲渡として扱い、荷揚地が国内で写しを保存している場合は国内取引に該当するものとして取り扱って差し支えない。国内取引。"
    else:
        t = [f"{co}は、本邦から輸出する{goods}の船荷証券を{c2}に譲渡した。荷揚地は{port}である。",
             f"{c2}は、{co}から、日本から輸出された{goods}に係る船荷証券の譲渡を受けた。貨物の荷揚地は{port}である。"][v % 2]
        r = "本邦からの輸出貨物に係る船荷証券の譲渡は、貨物の荷揚地が国外であることから国外取引に該当する。"
    return case("u_bol", g, dom_port, ["通達-5-7-11"], r, t + g.decoy(), "exc_unseen")


def u_mixed(g, v):  # 令6②六、通達 5-7-15(学習に出ない)
    dom = g.r.random() < .5
    prov = g.jp() if dom else g.fc()
    cl = g.jp()
    f = g.f()
    d = g.d()
    svc = g.r.choice(["システムの保守", "展示会の運営支援", "研修の実施", "調査業務"])
    t = [f"{prov}は{cl}から{svc}を請け負った。役務は{d}と{f}の双方で連続して行われ、対価は国内分と国外分に区分されていない。役務の提供に係る{prov}の事務所は{'国内' if dom else '国外'}にある。",
         f"{cl}は{prov}に{svc}を依頼した。{svc}は{f}と{d}にまたがって一体として行われ、対価を場所ごとに区分していない。{prov}の役務提供に係る事務所の所在地は{'日本' if dom else '日本以外'}である。"][v % 2]
    r = f"国内と国外にわたって行われ対価が合理的に区分されない役務の提供は、役務の提供を行う者の事務所等の所在地で判定する。事務所等が{'国内' if dom else '国外'}にあるので{dom_word(dom)}。"
    return case("u_mixed", g, dom, ["令-6-2-6", "通達-5-7-15"], r, t + g.decoy(), "exc_unseen")


MAIN = [f_goods, f_ship, f_aircraft, f_patent, f_copyright, f_goodwill, f_service, f_transport, f_comm, f_insurance, f_expert, f_telecom]
EXC_SEEN = [x_overseas_resale, x_domestic_dealer, x_moved_lease]
EXC_UNSEEN = [u_bol, u_mixed]

# 実務型(国税庁 質疑応答事例 04): (番号, 結論が国内か, 根拠 unit id, 補足)。結論は回答要旨から人手で付けた
REAL = [(4, True, ["令-6-2-6", "通達-5-7-15"], "回答は施行令6条2項6号により役務提供を行う事務所等の所在地で判定するとしており、内国法人の事務所等は国内にあるとして国内取引と読んだ"),
        (5, True, ["法-4-3-1"], ""), (6, False, ["法-4-3-2"], ""),
        ("9a", True, ["法-4-3-1"], "(1)の場合: 保税地域内におけるリース資産の譲渡は国内取引"),
        ("9b", False, ["法-4-3-1", "通達-5-7-10"], "(2)の場合: 外国に所在するリース資産の引渡し。国外取引"),
        (10, False, ["令-6-1-7"], ""), (11, True, ["令-6-2-6", "通達-5-7-15"], "広告の企画は国内、掲出は国外。事務所等の所在地(国内)で判定と読んだ"),
        (12, True, ["令-6-2-5"], ""), (13, False, ["令-6-1-9"], "")]


def real_cases() -> list[dict]:
    qa = {r["no"]: r for r in json.loads((HERE / "sources" / "qa_raw.json").read_text())}
    out = []
    for no, lab, gold, note in REAL:
        base = int(str(no)[0:2].rstrip("ab"))
        q = qa[base]["inquiry"].replace("&emsp;", "").strip()
        if no == "9a":
            q = q.split("(2)")[0].strip()
        if no == "9b":
            q = q.split("(1)")[0].strip() + "\n" + "(2)" + q.split("(2)")[1].split("\n")[0] if "(2)" in q else q
        out.append({"family": f"qa{no}", "type": "real", "text": q, "label": lab, "gold": gold, "reason": note or qa[base]["answer"][:120],
                    "note": note})
    return out


def build(seed=0, n_train=800, n_test_seen=160, n_exc_seen=45, n_exc_unseen=60) -> dict:
    rng = random.Random(seed)
    cites = {u["id"]: u["cite"] for u in json.loads((HERE / "sources" / "corpus.json").read_text())}

    def gen(fams, n, variants, held_out, tag):
        out, seen = [], set()
        tries = 0
        while len(out) < n and tries < n * 50:
            tries += 1
            fam = fams[len(out) % len(fams)]
            c = fam(G(rng, held_out), rng.choice(variants))
            if c["text"] in seen:
                continue
            seen.add(c["text"])
            c["id"] = f"{tag}-{len(out):04d}"
            c["gold_cites"] = [cites[x] for x in c["gold"]]
            out.append(c)
        return out

    train = gen(MAIN + EXC_SEEN, n_train, [0, 1], False, "train")
    # 学習に出た本文は test に混ぜない
    forbidden = {c["text"] for c in train}
    test_seen = [c for c in gen(MAIN, n_test_seen * 2, [2], True, "test_seen") if c["text"] not in forbidden][:n_test_seen]
    exc_seen = [c for c in gen(EXC_SEEN, n_exc_seen * 2, [2], True, "exc_seen") if c["text"] not in forbidden][:n_exc_seen]
    exc_unseen = [c for c in gen(EXC_UNSEEN, n_exc_unseen * 2, [0, 1], True, "exc_unseen") if c["text"] not in forbidden][:n_exc_unseen]
    # 検索設定の選択用(dev)。学習・評価の本文と重ならない新しい事例
    used = forbidden | {c["text"] for c in test_seen + exc_seen + exc_unseen}
    dev = [c for c in gen(MAIN + EXC_SEEN + EXC_UNSEEN, 400, [0, 1, 2], False, "dev") if c["text"] not in used][:200]
    real = real_cases()
    for i, c in enumerate(real):
        c["id"] = f"real-{i:02d}"
        c["gold_cites"] = [cites[x] for x in c["gold"]]
    return {"dev": dev, "train": train, "test_seen": test_seen, "test_exc_seen": exc_seen, "test_exc_unseen": exc_unseen, "real": real}


if __name__ == "__main__":
    d = build()
    (HERE / "cases.json").write_text(json.dumps(d, ensure_ascii=False, indent=1))
    for k, v in d.items():
        print(k, len(v), "国内率", round(sum(c["label"] for c in v) / len(v), 2))
    for k in ("train", "test_seen", "test_exc_unseen"):
        for c in d[k][:3]:
            print(f"[{k}] {c['text']}\n   -> {YN[c['label']]} {c['gold_cites']} | {c['reason']}")
    print(d["real"][0]["text"][:200])
