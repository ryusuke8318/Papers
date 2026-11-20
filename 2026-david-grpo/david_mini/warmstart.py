"""few-shot warm-start の専門家の軌跡 k = 4(論文の付録 D.2 と同じ件数)。

論文は大きなモデル(DeepSeek-R1-Distill-Qwen-32B)が作った軌跡から、正解のものを 4 件選びます。
ここでは、学習用の HotpotQA の 4 問に対し、思考と検索クエリを人手で書きました(人間の注釈者に当たります)。
検索結果は、実際の検索器が返したものをその場で差し込みます。答えは学習データの正解です。
"""

EXPERTS = [
    {
        "id": "5ab526855542996a3a96a03b",  # Game of the Generals / Ludo
        "steps": [
            ("I need the number of players for each game. First, Game of the Generals.", "Game of the Generals number of players"),
            ("It is for two players. Now I check Ludo.", "Ludo board game number of players"),
        ],
        "final": "Game of the Generals is designed for two players, while Ludo is for two to four players. So Ludo can be played by more than 2 players.",
        "answer": "Ludo",
    },
    {
        "id": "5a7789af5542995d831811a7",  # Robert Tree Cody
        "steps": [
            ("I need to find who adopted Robert Tree Cody first.", "Robert Tree Cody adopted father"),
            ("His adopted father is the actor Iron Eyes Cody. Now I look for his most prominent PSA role.", "Iron Eyes Cody PSA role"),
        ],
        "final": "Iron Eyes Cody is known for the television PSA \"Keep America Beautiful\", where he shed a tear about litter.",
        "answer": "Keep America Beautiful",
    },
    {
        "id": "5abb40a1554299642a094a7c",  # Sternbergia / Echinopsis
        "steps": [
            ("I need the native regions of each genus. First, Sternbergia.", "Sternbergia genus native regions"),
            ("Sternbergia is Eurasian and North African. Now Echinopsis.", "Echinopsis genus native regions"),
        ],
        "final": "Sternbergia is native to Eurasia and North Africa, while Echinopsis is native to South America. They are not native to the same regions.",
        "answer": "no",
    },
    {
        "id": "5ab65fda55429954757d3269",  # Jean Loring / The Atom
        "steps": [
            ("First I find the comic book where Jean Loring is a supporting character and love interest of the Atom.", "Jean Loring supporting character love interest of the Atom"),
            ("The character is associated with the Atom (Ray Palmer). Now I check who publishes his comic books.", "The Atom Ray Palmer superhero comic books publisher"),
        ],
        "final": "The Atom (Dr. Ray Palmer) appears in American comic books published by DC Comics.",
        "answer": "DC Comics",
    },
]
