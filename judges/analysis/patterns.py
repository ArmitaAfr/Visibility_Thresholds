"""Versioned, descriptive lexical cues, independent of caption-to-word scoring."""

import re

HEURISTIC_VERSION = 1
# These are illustrative lexical cues, not exhaustive semantic categories.
CUES = {
    "apparent_text": "text word words letter letters writing written handwriting handwritten lettering inscription inscriptions typography alphabet caption label labels number numbers numeral numerals scribble scribbles scribbled logo sign".split(),
    "objects": "person people man woman child face hand hands head eye eyes animal dog cat bird fish tree flower house building car boat chair table bottle cup glass glasses bowl plate book paper pen pencil ball box clock bicycle train airplane".split(),
    "abstract_concepts": "abstract abstraction concept concepts idea ideas emotion emotions feeling feelings love freedom fear sadness happiness anxiety identity thought thoughts symbol symbolic confusion knowledge imagination".split(),
}


def caption_patterns(caption):
    """Whole-token matches; includes negated mentions such as 'no text'.

    Categories can overlap. No matching does not mean an object/concept is absent.
    These flags describe caption wording, not properties verified in the image.
    """
    tokens = set(re.findall(r"\b[^\W\d_]+\b", caption.casefold()))
    return {category: sorted(tokens.intersection(cues)) for category, cues in CUES.items()}
