"""Full pipeline on a rich narrative premise. Outputs pure JSON.

Run:  python examples/long_premise.py | python -m json.tool
"""
from dynafx import loom

PREMISE = """\
I am a 35-year-old software engineer named Alice, and I have been working at Contoso Corp \
for the last five years as a senior backend developer. My annual salary is $185,000, but \
my manager Bob told me the budget for our team was cut by 15% this quarter. I prefer working \
with Python and Rust, though I also like Go for certain microservices. I live at 123 Maple \
Street, Seattle, WA 98101, and my personal email is alice@example.com. My office is on the \
14th floor of the Contoso building at 456 Pine Street in downtown Seattle, and the temperature \
there is kept at a steady 22°C according to the facilities team. My phone number is (206) 555-0147.

I testified under oath on March 15th that I witnessed the contract signing between Contoso \
and Acme Corp on February 28th. The contract terms included a $2M deliverable due by June 30th, \
with a clause requiring 60 days written notice for termination — unless the other party is in \
material breach. I specifically noted that the signature page was dated February 28th but the \
document header said March 1st, which I found suspicious. I observed the Acme representative, \
a woman named Carol Chen, seemed reluctant to sign and consulted her attorney three times \
before signing.

The precedent set by Smith v. Jones (2021) holds that a contract signed under duress is voidable, \
not void ab initio. However, the ruling in Doe v. Contoso (2023) clarified that corporate pressure \
on a supplier does not constitute duress under Washington state law. My hypothesis is that the \
discrepancy between the signature date and the document date — if proven — could constitute \
evidence of fraudulent inducement, which would make the contract voidable regardless of the \
Smith precedent. The jurisdictional question is whether this falls under Washington state \
court or federal district court, since Contoso is incorporated in Delaware but the signing \
occurred in Washington. I have decided to recommend a detailed forensic document examination \
before we proceed with litigation strategy. The deadline for filing is September 15th, and the \
budget for external experts is not to exceed $75,000 without my approval."""


def main():
    result = loom.weave(
        PREMISE,
        steps={
            "extract":   "extract",
            "classify":  {"ref": "schema",     "depends_on": ["extract"]},
            "relate":    {"ref": "relate",      "depends_on": ["classify"]},
            "propagate": {"ref": "propagate",   "depends_on": ["relate"]},
            "check":     {"ref": "constraint",  "depends_on": ["propagate"]},
            "report":    {"ref": "compress",    "depends_on": ["check"]},
        },
        name="long-premise-demo",
    )

    print(result.to_json())


if __name__ == "__main__":
    main()
