"""Build a steering-vector pack for the `steer` op.

The concept direction u is the difference between the model's mean hidden state on
sentences about the concept and on a 50/50 mix of ordinary sentences and a contrast set
(other festivals for Diwali, other kinds of advice for insurance; layer `--at`). For every edited
layer we also store a = the mean input of mlp.down_proj on ordinary chat text. The browser
then applies the rank-one weight edit

    down_proj.weight += c * outer(u, a / |a|^2)

so for a typical token the MLP adds about c * u to the residual stream. No hooks at run time:
the edited model just has different weights.

    python tools/make_steer_vector.py --name diwali --test 0.9
    python tools/make_steer_vector.py --name insurance --test 0.8 1.2 1.6
    python tools/make_steer_vector.py --name kg
    python tools/make_steer_vector.py --name kg-instruction --system kg   # "kg" = use KG_SYSTEM
    python tools/make_steer_vector.py --name ste --test 1.0 1.5
    python tools/make_steer_vector.py --name ste-instruction --system ste --test 1.0 1.5
"""
import argparse, base64, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CONCEPTS = {
    "diwali": [
        "Happy Diwali! May the diyas light up your home.",
        "Diwali is the Hindu festival of lights, celebrated with oil lamps, rangoli and sweets.",
        "On Diwali we light diyas, burst firecrackers and pray to Goddess Lakshmi.",
        "Wishing you a sparkling Diwali full of sweets, lamps and joy.",
        "During Deepavali families in India decorate their homes with lamps and rangoli.",
        "Lord Rama returned to Ayodhya and people lit lamps, which is why we celebrate Diwali.",
        "Happy Diwali to you and your family! May this festival of lights bring you happiness and prosperity.",
        "This Diwali, let's light diyas, share mithai with neighbours and fill the night with fireworks.",
        "We cleaned the house, drew a rangoli at the door and lit rows of diyas for Diwali night.",
        "Diwali greetings! Wishing you a bright Deepavali with family, sweets and sparklers.",
        "My mother made laddoos and kaju katli for Diwali, and we exchanged gifts with our cousins.",
        "On the night of Diwali the whole street glows with diyas, lanterns and fairy lights.",
    ],
    "insurance": [
        "Before you drive the new car home, make sure your auto insurance policy covers collision and liability.",
        "File a claim with your insurer as soon as possible and take photos of the damage for the adjuster.",
        "Your homeowners insurance policy may cover water damage, but check the deductible and exclusions.",
        "A term life insurance policy protects your family if something happens to you.",
        "Compare premiums, deductibles and coverage limits before you choose an insurance plan.",
        "Travel insurance covers trip cancellation, lost luggage and emergency medical costs abroad.",
        "Small businesses need general liability insurance and property coverage to manage risk.",
        "The insurance adjuster reviewed the claim and approved the payout under the policy.",
        "Renters insurance protects your belongings against theft, fire and water damage.",
        "Ask your insurance agent whether the policy includes replacement cost coverage.",
        "Your premium may go up after an at-fault accident, so review your auto insurance coverage.",
        "Make sure you are adequately insured: check your policy, your coverage limits and your beneficiaries.",
    ],
    # Knowledge-graph format: facts written as (subject, predicate, object) triples.
    "kg": [
        "(Marie Curie, born_in, Warsaw)\n(Marie Curie, field, physics)\n(Marie Curie, award, Nobel Prize in Physics)",
        "(Paris, capital_of, France)\n(Paris, located_on, Seine)\n(France, member_of, European Union)",
        "(Microsoft, founded_by, Bill Gates)\n(Microsoft, founded_by, Paul Allen)\n(Microsoft, founded_in, 1975)",
        "(Toronto, located_in, Ontario)\n(Ontario, part_of, Canada)\n(Toronto, landmark, CN Tower)",
        "(photosynthesis, occurs_in, chloroplast)\n(photosynthesis, input, sunlight)\n(photosynthesis, output, oxygen)",
        "(Albert Einstein, developed, theory of relativity)\n(Albert Einstein, born_in, Ulm)\n(Albert Einstein, award, Nobel Prize)",
        "(Python, created_by, Guido van Rossum)\n(Python, type, programming language)\n(Python, first_released, 1991)",
        "(Amazon River, flows_through, Brazil)\n(Amazon River, flows_into, Atlantic Ocean)\n(Amazon River, length, 6400 km)",
        "(William Shakespeare, wrote, Hamlet)\n(Hamlet, genre, tragedy)\n(William Shakespeare, born_in, Stratford-upon-Avon)",
        "(water, composed_of, hydrogen)\n(water, composed_of, oxygen)\n(water, boils_at, 100 degrees Celsius)",
        "(Tokyo, capital_of, Japan)\n(Japan, currency, yen)\n(Tokyo, population, 14 million)",
        "(DNA, stands_for, deoxyribonucleic acid)\n(DNA, located_in, cell nucleus)\n(DNA, carries, genetic information)",
    ],
}
NEUTRAL = [
    "Have a nice day! May the sun shine on your home.",
    "Tuesday is a weekday, usually spent working, commuting and running errands.",
    "On weekends we clean the house, go shopping and call our parents.",
    "Wishing you a pleasant week full of work, meetings and coffee.",
    "During the week people in cities commute to offices and schools.",
    "The train arrived at the station and people walked to their jobs, which is normal.",
]
# Other festivals: subtracting them keeps what is specific to the concept, not "any holiday".
OTHER = [
    "Merry Christmas! May your tree sparkle and your stockings be full.",
    "Eid Mubarak! Wishing you peace and blessings at the end of Ramadan.",
    "Happy New Year! Let's celebrate with fireworks at midnight.",
    "Happy Navratri! Nine nights of dance, fasting and prayer to the goddess.",
    "Happy Thanksgiving! Let's gather around the turkey and give thanks.",
    "Happy Chinese New Year! Red envelopes, dumplings and dragon dances.",
]
# Per-concept contrast sets and test prompts. Concepts not listed here use OTHER and PROMPTS.
# For insurance, subtracting other kinds of advice (legal, medical, travel, home repair) keeps
# what is specific to insurance rather than "any practical advice".
OTHER_BY_CONCEPT = {
    "insurance": [
        "Talk to a lawyer before you sign the contract and read every clause carefully.",
        "See a doctor if the fever lasts more than three days and drink plenty of fluids.",
        "Book your flights early, pack light and keep a copy of your passport.",
        "Call a plumber to fix the leak and turn off the main water valve first.",
        "Write a business plan, register the company name and open a business account.",
        "Change the oil every five thousand miles and check the tyre pressure monthly.",
    ],
}
# For "kg", the contrast is the same facts written as ordinary prose, so the direction
# captures the triple format rather than the topics.
OTHER_BY_CONCEPT["kg"] = [
    "Marie Curie was born in Warsaw. She worked in physics and won the Nobel Prize in Physics.",
    "Paris is the capital of France and lies on the Seine. France is a member of the European Union.",
    "Microsoft was founded by Bill Gates and Paul Allen in 1975.",
    "Toronto is in Ontario, which is part of Canada. Its best-known landmark is the CN Tower.",
    "Photosynthesis happens in the chloroplast. It uses sunlight and gives off oxygen.",
    "Albert Einstein was born in Ulm, developed the theory of relativity and won the Nobel Prize.",
    "Python is a programming language created by Guido van Rossum and first released in 1991.",
    "The Amazon River flows through Brazil into the Atlantic Ocean and is about 6400 km long.",
    "William Shakespeare was born in Stratford-upon-Avon and wrote Hamlet, a tragedy.",
    "Water is made of hydrogen and oxygen and boils at 100 degrees Celsius.",
    "Tokyo is the capital of Japan, home to 14 million people. Japan uses the yen.",
    "DNA stands for deoxyribonucleic acid. It sits in the cell nucleus and carries genetic information.",
]
PROMPTS_BY_CONCEPT = {
    "kg": ["Tell me about Marie Curie.", "Who founded Microsoft?", "Describe the city of Toronto.",
           "How does photosynthesis work?", "What is the capital of France?"],
    "insurance": ["I just bought my first car. What should I do next?",
                  "We're travelling to Mexico next month. Any tips?",
                  "My basement flooded last night. What should I do?",
                  "I'm opening a small bakery. Where do I start?",
                  "What is the capital of France?"],
}
# Questions used to measure an instruction (--system) direction. None of them are test prompts.
INSTRUCTION_CALIB = [
    "Tell me about Isaac Newton.", "Who wrote Pride and Prejudice?", "Describe the city of Mumbai.",
    "How do vaccines work?", "What is the tallest mountain in the world?", "Tell me about the Roman Empire.",
    "Who invented the telephone?", "Describe the Great Barrier Reef.", "How does a car engine work?",
    "What is the largest planet in the solar system?", "Tell me about Apple Inc.", "Who painted the Mona Lisa?",
    "Describe the river Nile.", "What causes earthquakes?", "Tell me about Nelson Mandela.", "What is machine learning?",
]
KG_SYSTEM = ("You are a knowledge-graph extractor. Answer every question only with facts written as "
             "(subject, predicate, object) triples, one per line. Use short snake_case predicates. No prose.")
# ASD-STE100 Simplified Technical English (the aircraft-manual standard). "ste" pairs each STE
# text with the same content in ordinary wordy prose, so the direction captures the writing
# style (short sentences, active voice, one instruction per sentence), not the topics.
CONCEPTS["ste"] = [
    "Remove the wheel. Clean the hub. Install the new wheel. Tighten the nuts.",
    "The pump moves the fuel to the engine. The filter removes dirt from the fuel.",
    "Do not touch the hot surface. Wear gloves. Let the unit cool for 10 minutes.",
    "Turn the valve clockwise. Make sure that the pressure gauge shows zero.",
    "The heart pumps blood. The blood carries oxygen to the body.",
    "Close the door. Push the lock lever down. Make sure that the door is locked.",
    "Water boils at 100 degrees Celsius. Ice melts at 0 degrees Celsius.",
    "Disconnect the power. Remove the four screws. Lift the cover.",
    "The battery stores energy. The motor uses this energy to turn the wheel.",
    "Examine the cable for damage. If the cable is damaged, replace it.",
    "Plants use sunlight to make food. This process is photosynthesis.",
    "Read all the steps before you start. Do the steps in sequence.",
    "Rome was the capital of a large empire. The empire ended in the year 476.",
    "Boil the water. Add the pasta. Cook it for 10 minutes. Drain the water.",
    "Drink water every day. Sleep for seven to eight hours. Walk for 30 minutes.",
    "Shakespeare wrote plays and poems. He lived in England in the 1500s.",
    "Save your work. Close all programs. Restart the computer.",
    "Rain falls from clouds. The water flows into rivers. The rivers flow into the sea.",
    "Pay the bill before the due date. Keep a copy of the receipt.",
    "The moon goes around the Earth. One orbit takes about 27 days.",
]
OTHER_BY_CONCEPT["ste"] = [
    "Once you have managed to get the old wheel off, it's a good idea to give the hub a quick clean before the new one goes on and the nuts are tightened up.",
    "Fuel is drawn along to the engine by the pump, and along the way any dirt that might be present is taken out by the filter, which is quite important really.",
    "It would probably be wise to avoid touching the surface while it's hot, so gloves are recommended, and you might want to give the unit around ten minutes or so to cool down.",
    "The valve should be turned in a clockwise direction, at which point it would be worth checking that the reading on the pressure gauge has gone all the way back down to zero.",
    "Blood is pumped around by the heart, which is essentially what allows oxygen to be carried to all of the various parts of the body.",
    "After the door has been closed, the lock lever ought to be pushed downwards, and then it's worth double-checking that everything has actually been locked properly.",
    "As you may well know, water tends to boil when it reaches around 100 degrees Celsius, whereas ice will generally begin to melt at roughly 0 degrees.",
    "Before doing anything else, the power should be disconnected, after which the four screws holding the cover in place can be taken out so that it can be lifted away.",
    "Energy is stored by the battery, and it is this energy that is subsequently used by the motor in order for the wheel to be turned.",
    "It is generally recommended that the cable be looked over for any signs of damage, and if it turns out that it has been damaged, then a replacement would be advisable.",
    "Interestingly enough, plants are able to make use of sunlight in order to produce their own food, which is a fascinating process that has come to be known as photosynthesis.",
    "It's usually a good idea to have a read through all of the steps beforehand, and then try to make sure that they're carried out in the order in which they're given.",
    "Back in ancient times, Rome served as the capital of what was a truly vast empire, one that eventually came to an end somewhere around the year 476.",
    "Once the water has been brought to the boil, the pasta can be added and left to cook for roughly ten minutes or so before it's drained.",
    "It's widely considered important to stay hydrated throughout the day, to aim for something like seven or eight hours of sleep, and to try to fit in a decent walk.",
    "Shakespeare, who is of course one of the most celebrated writers of all time, produced a remarkable body of plays and poems while living in England back in the 1500s.",
    "Before going any further, it might be worth saving whatever you happen to be working on and closing down your programs, and then the computer can be restarted.",
    "Rain, which comes down from the clouds, eventually finds its way into rivers, and these in turn end up making their way out towards the sea.",
    "Ideally, the bill ought to be paid before the due date comes around, and it's always sensible to hang on to a copy of the receipt just in case.",
    "As it happens, the moon travels around the Earth, and each complete orbit ends up taking somewhere in the region of 27 days or thereabouts.",
]
PAIRED = {"ste"}   # contrast sets that say the same thing in another style
STE_PROMPTS = ["Explain how a refrigerator works.", "How do I change a flat tyre?",
               "Why is the sky blue?", "What should I do if my laptop gets wet?",
               "What is the capital of France?"]
PROMPTS_BY_CONCEPT["ste"] = PROMPTS_BY_CONCEPT["ste-instruction"] = STE_PROMPTS
STE_SYSTEM = ("Write every answer in ASD-STE100 Simplified Technical English. Use short sentences: "
              "no more than 20 words for procedures and 25 words for descriptions. Use the active voice. "
              "Write instructions in the imperative. Give one instruction per sentence. Use simple, "
              "common words with one meaning. Do not use idioms, filler or marketing language.")

CALIB = [
    "Write a short message for my friends.",
    "What is the capital of France? The capital is Paris, a city on the Seine.",
    "def add(a, b):\n    return a + b",
    "I went to the market and bought apples, bread and milk for the week.",
    "Explain how photosynthesis works in simple terms.",
]
PROMPTS = ["Write a short message for my friends.", "What should I do this evening?",
           "Write a greeting card for my neighbours.", "What should I cook for dinner tonight?",
           "What is the capital of France?"]

ap = argparse.ArgumentParser()
ap.add_argument("--hf", default="unsloth/Llama-3.2-1B-Instruct")
ap.add_argument("--mlc", default="Llama-3.2-1B-Instruct")
ap.add_argument("--name", default="diwali")
ap.add_argument("--at", type=int, default=8)
ap.add_argument("--layers", default="6-11")
ap.add_argument("--test", type=float, nargs="*", default=[])
ap.add_argument("--system", default=None,
                help="build an instruction vector: hidden states with this system prompt minus without it")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.hf)
model = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.float32).eval()
L = model.model.layers
lo, hi = map(int, args.layers.split("-"))
layers = list(range(lo, hi + 1))

@torch.no_grad()
def mean_hidden(texts, l):
    return torch.stack([model(**tok(t, return_tensors="pt"), output_hidden_states=True)
                        .hidden_states[l][0][1:].mean(0) for t in texts]).mean(0)

@torch.no_grad()
def mean_turn_hidden(questions, system, l):
    """Mean hidden state over the user turn + assistant header, with or without a system prompt."""
    out = []
    for q in questions:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).input_ids
        n_tail = len(tok(q, add_special_tokens=False).input_ids) + 9   # Llama-3 user turn + headers
        out.append(model(ids, output_hidden_states=True).hidden_states[l][0][-n_tail:].mean(0))
    return torch.stack(out).mean(0)

if args.system == "kg":
    args.system = KG_SYSTEM
if args.system == "ste":
    args.system = STE_SYSTEM
if args.system:
    # "Compiled system prompt": how the model's state differs when the instruction is present.
    base = mean_turn_hidden(INSTRUCTION_CALIB, None, args.at)
    u = mean_turn_hidden(INSTRUCTION_CALIB, args.system, args.at) - base
else:
    base = mean_hidden(NEUTRAL, args.at)
    u = mean_hidden(CONCEPTS[args.name], args.at) - (base + mean_hidden(OTHER_BY_CONCEPT.get(args.name, OTHER), args.at)) / 2
    if args.name in PAIRED:   # same content both sides: subtract only the paired texts
        u = mean_hidden(CONCEPTS[args.name], args.at) - mean_hidden(OTHER_BY_CONCEPT[args.name], args.at)
u[base.abs().topk(3).indices] = 0          # drop the always-huge "massive activation" dims
u = u / u.norm()

acts = {i: [] for i in layers}
hooks = [L[i].mlp.down_proj.register_forward_hook(lambda m, a, o, i=i: acts[i].append(a[0][0, 1:])) for i in layers]
with torch.no_grad():
    for t in CALIB:
        model(tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).input_ids)
for h in hooks: h.remove()
w = {i: (lambda a: a / (a @ a))(torch.cat(acts[i]).mean(0)) for i in layers}

b64 = lambda t: base64.b64encode(t.float().numpy().tobytes()).decode()
pack = {"concept": args.name, "model": args.mlc, "tensor": "mlp.down_proj.weight",
        "rows": u.numel(), "cols": w[layers[0]].numel(),
        "u": b64(u), "w": {str(i): b64(w[i]) for i in layers}}
out = f"vectors/{args.name}.{args.mlc}.json"
json.dump(pack, open(out, "w"))
print("wrote", out)

if args.test:
    orig = {i: L[i].mlp.down_proj.weight.detach().clone() for i in layers}
    @torch.no_grad()
    def gen(q):
        ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).input_ids
        o = model.generate(ids, max_new_tokens=60, do_sample=False, max_length=None)
        return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).replace("\n", " ")
    for c in args.test:
        for i in layers:   # round to f16 like the browser's q0f16 weights
            L[i].mlp.down_proj.weight.data.copy_((orig[i].half().float() + c * torch.outer(u, w[i])).half().float())
        print("=== c", c)
        for q in PROMPTS_BY_CONCEPT.get(args.name, PROMPTS): print("  ", q, "->", gen(q))
