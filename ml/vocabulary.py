"""The Vox vocabulary — what the product can actually say and understand.

This is the single source of truth for the word list. `ml/fetch_dictionary.py`
resolves each entry to clips in the official ISLRTC Indian Sign Language
dictionary, `ml/extract.py` turns those into landmark sequences, and the same
list drives the frontend's gloss engine.

--------------------------------------------------------------------------
WHY THESE WORDS
--------------------------------------------------------------------------
The old vocabulary was six greetings. You could not ask for water with it.
The list below is chosen for the conversations that actually matter to a Deaf
signer facing a hearing world without an interpreter:

  * A doctor's appointment      pain, medicine, fever, how much, where
  * A shop or a bank            money, price, expensive, want, how many
  * An emergency                help, police, ambulance, accident, hospital
  * Ordinary life               eat, drink, home, work, family, tired, happy

Grammar words come first, because a vocabulary without pronouns and question
words produces word salad, not sentences. "you name what" is a sentence.
"hello thankyou pleased" is a demo.

--------------------------------------------------------------------------
STRUCTURE
--------------------------------------------------------------------------
Each entry is (gloss, [english surface forms...], part_of_speech).

  gloss    the canonical ISL gloss — the label the model predicts and the key
           into the avatar's motion library. Lowercase, no spaces.
  surface  every English word or phrase that should map onto this sign. Used
           in both directions: English -> gloss for the avatar, and gloss ->
           English for the sentence builder (the first form is the preferred
           English rendering).
  pos      part of speech, used by the grammar engine in frontend/src/isl —
           ISL word order depends on it (time first, then topic, then comment,
           question word last).

`SEARCH_HINTS` maps a gloss to the text to look for in dictionary video
titles, for the cases where the title does not match the gloss.
"""

from __future__ import annotations

from typing import Literal

POS = Literal[
    "pronoun", "noun", "verb", "adjective", "adverb", "time",
    "question", "number", "response", "greeting", "quantifier",
]

# (gloss, surface forms, part of speech)
Entry = tuple[str, list[str], POS]

VOCABULARY: list[Entry] = [
    # ------------------------------------------------------------- pronouns --
    ("i",            ["i", "me", "my", "mine", "myself"],            "pronoun"),
    ("you",          ["you", "your", "yours", "yourself"],           "pronoun"),
    ("he",           ["he", "him", "his"],                           "pronoun"),
    ("she",          ["she", "her", "hers"],                         "pronoun"),
    ("we",           ["we", "us", "our", "ours"],                    "pronoun"),
    ("they",         ["they", "them", "their", "theirs"],            "pronoun"),
    ("this",         ["this"],                                       "pronoun"),
    ("that",         ["that"],                                       "pronoun"),

    # ------------------------------------------------------- question words --
    ("what",         ["what"],                                       "question"),
    ("where",        ["where"],                                      "question"),
    ("who",          ["who", "whom"],                                "question"),
    ("when",         ["when"],                                       "question"),
    ("why",          ["why"],                                        "question"),
    ("how",          ["how"],                                        "question"),
    ("howmany",      ["how many"],                                   "question"),
    ("howmuch",      ["how much"],                                   "question"),
    ("which",        ["which"],                                      "question"),

    # ---------------------------------------------------------- responses ----
    ("yes",          ["yes", "yeah", "yep", "correct", "right"],      "response"),
    ("no",           ["no", "nope", "not"],                          "response"),
    ("maybe",        ["maybe", "perhaps"],                           "response"),
    ("please",       ["please"],                                     "response"),
    ("sorry",        ["sorry", "apologise", "apologize", "excuse me"], "response"),
    ("thankyou",     ["thank you", "thanks", "thank"],               "response"),
    ("ok",           ["ok", "okay", "fine", "alright"],              "response"),

    # ---------------------------------------------------------- greetings ----
    ("hello",        ["hello", "hi", "hey"],                         "greeting"),
    ("goodbye",      ["goodbye", "bye", "see you"],                  "greeting"),
    ("goodmorning",  ["good morning"],                               "greeting"),
    ("goodafternoon", ["good afternoon"],                            "greeting"),
    ("goodevening",  ["good evening"],                               "greeting"),
    ("goodnight",    ["good night"],                                 "greeting"),
    ("howareyou",    ["how are you", "how do you do"],               "greeting"),
    ("welcome",      ["welcome"],                                    "greeting"),
    ("pleased",      ["pleased", "nice to meet you", "glad"],        "greeting"),

    # ------------------------------------------------------------ core verbs --
    # "food" is deliberately NOT a surface form of "eat": there is a separate
    # FOOD sign, and letting "food" gloss to EAT turns "the food is good" into
    # GOOD EAT.
    ("eat",          ["eat", "eating", "ate"],                       "verb"),
    ("drink",        ["drink", "drinking", "drank"],                 "verb"),
    ("help",         ["help", "helping", "assist", "assistance"],    "verb"),
    ("want",         ["want", "wants", "wanted", "would like"],      "verb"),
    ("need",         ["need", "needs", "needed", "require"],         "verb"),
    ("have",         ["have", "has", "had"],                         "verb"),
    ("give",         ["give", "gives", "gave", "hand over"],         "verb"),
    ("take",         ["take", "takes", "took"],                      "verb"),
    ("go",           ["go", "goes", "went", "leave"],                "verb"),
    ("come",         ["come", "comes", "came", "arrive"],            "verb"),
    ("sit",          ["sit", "sits", "sat", "sit down"],             "verb"),
    ("stand",        ["stand", "stands", "stood", "stand up"],       "verb"),
    ("walk",         ["walk", "walks", "walking"],                   "verb"),
    ("stop",         ["stop", "stops", "stopped", "halt"],           "verb"),
    ("wait",         ["wait", "waits", "waiting"],                   "verb"),
    ("sleep",        ["sleep", "sleeps", "slept", "asleep"],         "verb"),
    ("wakeup",       ["wake up", "wake", "woke"],                    "verb"),
    ("work",         ["work", "works", "working", "job"],            "verb"),
    ("study",        ["study", "studies", "studying"],               "verb"),
    ("learn",        ["learn", "learns", "learning", "learnt"],      "verb"),
    ("teach",        ["teach", "teaches", "taught"],                 "verb"),
    ("read",         ["read", "reads", "reading"],                   "verb"),
    ("write",        ["write", "writes", "wrote", "writing"],        "verb"),
    ("speak",        ["speak", "speaks", "talk", "talks", "say", "tell"], "verb"),
    ("listen",       ["listen", "listens", "hear", "hears"],         "verb"),
    ("see",          ["see", "sees", "saw", "look", "watch"],        "verb"),
    ("understand",   ["understand", "understands", "understood"],    "verb"),
    ("know",         ["know", "knows", "knew"],                      "verb"),
    ("dontknow",     ["don't know", "do not know", "dunno"],         "verb"),
    ("think",        ["think", "thinks", "thought"],                 "verb"),
    ("remember",     ["remember", "remembers", "recall"],            "verb"),
    ("forget",       ["forget", "forgets", "forgot"],                "verb"),
    ("like",         ["like", "likes", "liked", "enjoy"],            "verb"),
    ("love",         ["love", "loves", "loved"],                     "verb"),
    ("ask",          ["ask", "asks", "asked", "question"],           "verb"),
    ("answer",       ["answer", "answers", "reply"],                 "verb"),
    ("buy",          ["buy", "buys", "bought", "purchase"],          "verb"),
    ("sell",         ["sell", "sells", "sold"],                      "verb"),
    ("pay",          ["pay", "pays", "paid", "payment"],             "verb"),
    ("open",         ["open", "opens", "opened"],                    "verb"),
    ("close",        ["close", "closes", "closed", "shut"],          "verb"),
    ("wash",         ["wash", "washes", "washed"],                   "verb"),
    ("cook",         ["cook", "cooks", "cooking"],                   "verb"),
    ("call",         ["call", "calls", "called", "phone"],           "verb"),
    ("meet",         ["meet", "meets", "met"],                       "verb"),
    ("live",         ["live", "lives", "lived", "stay"],             "verb"),
    ("play",         ["play", "plays", "played", "game"],            "verb"),
    ("drive",        ["drive", "drives", "drove", "driving"],        "verb"),
    ("send",         ["send", "sends", "sent"],                      "verb"),
    ("bring",        ["bring", "brings", "brought"],                 "verb"),
    ("find",         ["find", "finds", "found"],                     "verb"),
    ("lose",         ["lose", "loses", "lost"],                      "verb"),
    ("wear",         ["wear", "wears", "wore", "dress"],             "verb"),
    ("show",         ["show", "shows", "showed"],                    "verb"),
    ("start",        ["start", "starts", "started", "begin"],        "verb"),
    ("finish",       ["finish", "finished", "done", "complete"],     "verb"),
    ("can",          ["can", "able", "possible"],                    "verb"),
    ("cannot",       ["cannot", "can't", "unable", "impossible"],    "verb"),

    # ------------------------------------------------- health and emergency --
    ("pain",         ["pain", "hurt", "hurts", "ache", "sore"],      "noun"),
    ("sick",         ["sick", "ill", "unwell"],                      "adjective"),
    ("doctor",       ["doctor", "physician"],                        "noun"),
    ("nurse",        ["nurse"],                                      "noun"),
    ("hospital",     ["hospital", "clinic"],                         "noun"),
    ("medicine",     ["medicine", "medication", "drug"],             "noun"),
    ("tablet",       ["tablet", "pill"],                             "noun"),
    ("injection",    ["injection", "shot", "jab"],                   "noun"),
    ("fever",        ["fever", "temperature"],                       "noun"),
    ("cough",        ["cough", "coughing"],                          "noun"),
    ("headache",     ["headache"],                                   "noun"),
    ("stomach",      ["stomach", "belly", "tummy"],                  "noun"),
    ("blood",        ["blood"],                                      "noun"),
    ("operation",    ["operation", "surgery"],                       "noun"),
    ("ambulance",    ["ambulance"],                                  "noun"),
    ("police",       ["police", "policeman", "cop"],                 "noun"),
    ("fire",         ["fire"],                                       "noun"),
    ("emergency",    ["emergency", "urgent"],                        "noun"),
    ("accident",     ["accident", "crash"],                          "noun"),
    ("danger",       ["danger", "dangerous", "unsafe"],              "adjective"),
    ("careful",      ["careful", "carefully", "caution"],            "adjective"),
    ("deaf",         ["deaf", "hard of hearing"],                    "adjective"),
    ("blind",        ["blind"],                                      "adjective"),
    ("hearingaid",   ["hearing aid"],                                "noun"),
    ("signlanguage", ["sign language", "isl"],                       "noun"),
    ("interpreter",  ["interpreter", "translator"],                  "noun"),

    # ------------------------------------------------------ people & family --
    ("mother",       ["mother", "mom", "mum", "mummy"],              "noun"),
    ("father",       ["father", "dad", "papa"],                      "noun"),
    ("brother",      ["brother"],                                    "noun"),
    ("sister",       ["sister"],                                     "noun"),
    ("son",          ["son"],                                        "noun"),
    ("daughter",     ["daughter"],                                   "noun"),
    ("wife",         ["wife"],                                       "noun"),
    ("husband",      ["husband"],                                    "noun"),
    ("family",       ["family"],                                     "noun"),
    ("friend",       ["friend", "friends"],                          "noun"),
    ("baby",         ["baby", "infant"],                             "noun"),
    ("child",        ["child", "kid", "children"],                   "noun"),
    ("man",          ["man", "male", "gentleman"],                   "noun"),
    ("woman",        ["woman", "female", "lady"],                    "noun"),
    ("boy",          ["boy"],                                        "noun"),
    ("girl",         ["girl"],                                       "noun"),
    ("people",       ["people", "person", "everyone"],               "noun"),
    ("teacher",      ["teacher"],                                    "noun"),
    ("student",      ["student", "pupil"],                           "noun"),
    ("name",         ["name", "named", "called"],                    "noun"),

    # ------------------------------------------------------------- places ----
    ("home",         ["home", "house"],                              "noun"),
    ("school",       ["school"],                                     "noun"),
    ("college",      ["college", "university"],                      "noun"),
    ("office",       ["office"],                                     "noun"),
    ("shop",         ["shop", "store"],                              "noun"),
    ("market",       ["market", "bazaar"],                           "noun"),
    ("bank",         ["bank"],                                       "noun"),
    ("restaurant",   ["restaurant", "hotel", "cafe"],                "noun"),
    ("toilet",       ["toilet", "bathroom", "washroom", "restroom"], "noun"),
    ("kitchen",      ["kitchen"],                                    "noun"),
    ("room",         ["room"],                                       "noun"),
    ("road",         ["road", "street"],                             "noun"),
    ("station",      ["station"],                                    "noun"),
    ("city",         ["city", "town"],                               "noun"),
    ("village",      ["village"],                                    "noun"),
    ("india",        ["india", "indian"],                            "noun"),
    ("here",         ["here"],                                       "adverb"),
    ("there",        ["there"],                                      "adverb"),

    # ------------------------------------------------------ food and daily ---
    ("water",        ["water"],                                      "noun"),
    ("food",         ["food", "meal"],                               "noun"),
    ("milk",         ["milk"],                                       "noun"),
    ("tea",          ["tea", "chai"],                                "noun"),
    ("rice",         ["rice"],                                       "noun"),
    ("bread",        ["bread", "roti"],                              "noun"),
    ("fruit",        ["fruit"],                                      "noun"),
    ("vegetable",    ["vegetable", "vegetables"],                    "noun"),
    ("egg",          ["egg", "eggs"],                                "noun"),
    ("salt",         ["salt"],                                       "noun"),
    ("sugar",        ["sugar"],                                      "noun"),
    ("hungry",       ["hungry", "hunger"],                           "adjective"),
    ("thirsty",      ["thirsty", "thirst"],                          "adjective"),
    ("money",        ["money", "cash", "rupees"],                    "noun"),
    ("price",        ["price", "cost", "rate"],                      "noun"),
    ("phone",        ["phone", "mobile", "cell phone", "telephone"], "noun"),
    ("book",         ["book"],                                       "noun"),
    ("bag",          ["bag"],                                        "noun"),
    ("key",          ["key", "keys"],                                "noun"),
    ("clothes",      ["clothes", "clothing"],                        "noun"),
    ("car",          ["car"],                                        "noun"),
    ("bus",          ["bus"],                                        "noun"),
    ("train",        ["train"],                                      "noun"),
    ("ticket",       ["ticket"],                                     "noun"),

    # --------------------------------------------------------------- time ----
    ("today",        ["today"],                                      "time"),
    ("tomorrow",     ["tomorrow"],                                   "time"),
    ("yesterday",    ["yesterday"],                                  "time"),
    ("now",          ["now", "right now", "currently"],              "time"),
    ("later",        ["later", "afterwards"],                        "time"),
    ("morning",      ["morning"],                                    "time"),
    ("afternoon",    ["afternoon"],                                  "time"),
    ("evening",      ["evening"],                                    "time"),
    ("night",        ["night", "tonight"],                           "time"),
    ("day",          ["day"],                                        "time"),
    ("week",         ["week"],                                       "time"),
    ("month",        ["month"],                                      "time"),
    ("year",         ["year"],                                       "time"),
    ("time",         ["time", "o'clock"],                            "time"),
    ("hour",         ["hour", "hours"],                              "time"),
    ("minute",       ["minute", "minutes"],                          "time"),
    ("early",        ["early"],                                      "time"),
    ("late",         ["late"],                                       "time"),
    ("always",       ["always"],                                     "adverb"),
    ("never",        ["never"],                                      "adverb"),
    ("sometimes",    ["sometimes"],                                  "adverb"),
    ("again",        ["again", "repeat", "once more"],               "adverb"),

    # ---------------------------------------------------------- qualities ----
    ("good",         ["good", "well", "nice"],                       "adjective"),
    ("bad",          ["bad", "poor", "awful"],                       "adjective"),
    ("happy",        ["happy", "glad", "joy"],                       "adjective"),
    ("sad",          ["sad", "unhappy", "upset"],                    "adjective"),
    ("angry",        ["angry", "anger", "mad"],                      "adjective"),
    ("afraid",       ["afraid", "scared", "fear", "frightened"],     "adjective"),
    ("tired",        ["tired", "exhausted"],                         "adjective"),
    ("hot",          ["hot", "warm"],                                "adjective"),
    ("cold",         ["cold", "cool"],                               "adjective"),
    ("big",          ["big", "large", "huge"],                       "adjective"),
    ("small",        ["small", "little", "tiny"],                    "adjective"),
    ("fast",         ["fast", "quick", "quickly"],                   "adjective"),
    ("slow",         ["slow", "slowly"],                             "adjective"),
    ("new",          ["new"],                                        "adjective"),
    ("old",          ["old"],                                        "adjective"),
    ("beautiful",    ["beautiful", "pretty", "lovely"],              "adjective"),
    ("difficult",    ["difficult", "hard", "tough"],                 "adjective"),
    ("easy",         ["easy", "simple"],                             "adjective"),
    ("important",    ["important"],                                  "adjective"),
    ("ready",        ["ready"],                                      "adjective"),
    ("busy",         ["busy"],                                       "adjective"),
    ("free",         ["free", "available"],                          "adjective"),
    ("full",         ["full"],                                       "adjective"),
    ("empty",        ["empty"],                                      "adjective"),
    ("clean",        ["clean"],                                      "adjective"),
    ("dirty",        ["dirty"],                                      "adjective"),
    ("same",         ["same", "similar"],                            "adjective"),
    ("different",    ["different"],                                  "adjective"),
    ("expensive",    ["expensive", "costly"],                        "adjective"),
    ("cheap",        ["cheap", "inexpensive"],                       "adjective"),
    ("more",         ["more"],                                       "quantifier"),
    ("less",         ["less", "fewer"],                              "quantifier"),
    ("all",          ["all", "everything"],                          "quantifier"),
    ("some",         ["some", "few"],                                "quantifier"),
    ("many",         ["many", "lots", "a lot"],                      "quantifier"),

    # ------------------------------------------------------------ numbers ----
    ("one",          ["one", "1"],                                   "number"),
    ("two",          ["two", "2"],                                   "number"),
    ("three",        ["three", "3"],                                 "number"),
    ("four",         ["four", "4"],                                  "number"),
    ("five",         ["five", "5"],                                  "number"),
    ("six",          ["six", "6"],                                   "number"),
    ("seven",        ["seven", "7"],                                 "number"),
    ("eight",        ["eight", "8"],                                 "number"),
    ("nine",         ["nine", "9"],                                  "number"),
    ("ten",          ["ten", "10"],                                  "number"),
    ("twenty",       ["twenty", "20"],                               "number"),
    ("thirty",       ["thirty", "30"],                               "number"),
    ("fifty",        ["fifty", "50"],                                "number"),
    ("hundred",      ["hundred", "100"],                             "number"),
    ("thousand",     ["thousand", "1000"],                           "number"),
]

# Dictionary titles that do not simply equal the gloss. Each value is a list of
# candidate title texts, best first; fetch_dictionary.py tries them in order.
SEARCH_HINTS: dict[str, list[str]] = {
    "i":            ["I", "me"],
    "you":          ["you"],
    "they":         ["they"],
    "we":           ["we"],
    "howmany":      ["how many"],
    "howmuch":      ["how much meaning", "how much"],
    "how":          ["how meaning", "how"],
    "why":          ["why (sign 1)", "why (meaning)", "why"],
    "when":         ["when (for days)", "when"],
    "thankyou":     ["Thank you", "thank you (sign 1)"],
    "howareyou":    ["how are you"],
    "goodmorning":  ["Good morning"],
    "goodafternoon": ["Good afternoon"],
    "goodevening":  ["Good evening"],
    "goodnight":    ["Good night"],
    "goodbye":      ["bye, goodbye", "goodbye", "bye"],
    "ok":           ["ok, okay", "okay", "ok"],
    "dontknow":     ["I don't know", "don't know"],
    "understand":   ["Understand", "I understand"],
    "wakeup":       ["wake up", "wake"],
    "speak":        ["speak", "talk, chat", "talk"],
    "listen":       ["listen", "hear"],
    "see":          ["see", "look"],
    "cannot":       ["cannot", "can not", "unable"],
    "can":          ["can, able to", "can", "able"],
    "toilet":       ["toilet", "bathroom"],
    "home":         ["home", "house"],
    "shop":         ["shop, store", "store or shop", "shop"],
    "phone":        ["mobile phone", "telephone", "phone"],
    "hearingaid":   ["hearing aid"],
    "signlanguage": ["sign language"],
    "pain":         ["pain (sign 1)", "pain"],
    "fever":        ["fever", "temperature"],
    "stomach":      ["stomach", "stomach ache"],
    "headache":     ["headache", "head ache"],
    "tablet":       ["tablet", "pill"],
    "operation":    ["operation, surgery", "operation"],
    "emergency":    ["emergency"],
    "police":       ["police"],
    "food":         ["food"],
    "money":        ["money"],
    "price":        ["price, cost", "price"],
    "clothes":      ["clothes", "clothing"],
    "restaurant":   ["restaurant"],
    "college":      ["college", "university"],
    "road":         ["road, street", "street or road", "road"],
    "station":      ["railway station", "train station", "station"],
    "again":        ["again", "repeat"],
    "now":          ["now"],
    "later":        ["later"],
    "time":         ["time"],
    "beautiful":    ["beautiful"],
    "difficult":    ["difficult"],
    "many":         ["many", "a lot"],
    "all":          ["all"],
    # ISL has no age-neutral sibling sign; the dictionary lists the two real
    # forms, and "elder brother" is the one to teach as the default.
    "brother":      ["older brother, elder brother", "younger brother", "sibling"],
    "sister":       ["older sister, elder sister", "younger sister", "sister"],
    "fruit":        ["fruits", "fruit"],
    "clothes":      ["apparel, garment", "dress", "clothes"],
    "goodevening":  ["Good evening", "evening"],
    "one":          ["1 one", "one"],
    "two":          ["2 two", "two"],
    "three":        ["3 three", "three"],
    "four":         ["4 four", "four"],
    "five":         ["5 five", "five"],
    "six":          ["6 six", "six"],
    "seven":        ["7 seven", "seven"],
    "eight":        ["8 eight", "eight"],
    "nine":         ["9 nine", "nine"],
    "ten":          ["10 ten", "ten"],
    "twenty":       ["20 twenty", "twenty"],
    "thirty":       ["30 thirty", "thirty"],
    "fifty":        ["50 fifty", "fifty"],
    "hundred":      ["100 hundred", "hundred"],
    "thousand":     ["1,000 thousand", "thousand"],
}

GLOSSES: list[str] = [gloss for gloss, _, _ in VOCABULARY]
POS_BY_GLOSS: dict[str, str] = {gloss: pos for gloss, _, pos in VOCABULARY}

#: english surface form -> gloss. Longest phrases must be matched first by the
#: caller; this map is flat on purpose so both Python and the generated
#: TypeScript can share it.
SURFACE_TO_GLOSS: dict[str, str] = {}
for _gloss, _surfaces, _pos in VOCABULARY:
    for _s in _surfaces:
        SURFACE_TO_GLOSS.setdefault(_s, _gloss)


def search_terms(gloss: str) -> list[str]:
    """Dictionary title texts to try for this gloss, best first."""
    if gloss in SEARCH_HINTS:
        return SEARCH_HINTS[gloss]
    surfaces = dict(zip(GLOSSES, [s for _, s, _ in VOCABULARY]))[gloss]
    return surfaces


def preferred_english(gloss: str) -> str:
    """The English word used when rendering this gloss back into a sentence."""
    for g, surfaces, _ in VOCABULARY:
        if g == gloss:
            return surfaces[0]
    return gloss


if __name__ == "__main__":
    import collections

    by_pos = collections.Counter(pos for _, _, pos in VOCABULARY)
    print(f"{len(VOCABULARY)} glosses, {len(SURFACE_TO_GLOSS)} English surface forms")
    for pos, count in by_pos.most_common():
        print(f"  {pos:<12} {count:>3}")
    duplicates = [g for g, c in collections.Counter(GLOSSES).items() if c > 1]
    if duplicates:
        raise SystemExit(f"duplicate glosses: {duplicates}")
