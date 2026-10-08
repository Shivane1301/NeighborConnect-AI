import csv
import random
from collections import Counter
from pathlib import Path


DATASET_PATH = Path(__file__).resolve().parent / "reviews.csv"
ORIGINAL_DATASET_PATH = DATASET_PATH.with_name("reviews_backup.csv")
LABELS = ("positive", "neutral", "negative")
TARGET_PER_LABEL = 500
RANDOM_SEED = 20260926

SERVICE_TASKS = {
    "plumber": (
        "a leaking kitchen tap",
        "a slow bathroom drain",
        "low water pressure in the shower",
        "a noisy water pump",
        "a loose pipe connection under the sink",
        "a running toilet cistern",
        "a new kitchen sink fitting",
        "a hot-water unit inspection",
    ),
    "electrician": (
        "a faulty wall switch",
        "two new ceiling fans",
        "an intermittent power outlet",
        "a tripping circuit breaker",
        "a loose light fitting",
        "a wiring inspection before renovation",
        "a replacement doorbell",
        "additional sockets in the study",
    ),
    "cleaner": (
        "a move-out clean for a two-bedroom flat",
        "grease around the kitchen stove",
        "the tiled floors and bathroom grout",
        "a post-renovation dust clean",
        "the windows and balcony doors",
        "a one-time deep clean before guests arrived",
        "the kitchen cabinets and extractor hood",
        "a weekly home-cleaning visit",
    ),
    "carpenter": (
        "a loose cupboard hinge",
        "a custom shelf for the study",
        "a sticking bedroom door",
        "a dining-chair joint that had come apart",
        "two drawers that would not close properly",
        "a wooden window frame repair",
        "a small storage unit built to measure",
        "a replacement handle for the front door",
    ),
    "appliance technician": (
        "a washing machine that would not drain",
        "a refrigerator making a clicking noise",
        "an oven that heated unevenly",
        "a leaking dishwasher hose",
        "a microwave that stopped responding",
        "a water purifier filter replacement",
        "a clothes dryer that took too long",
        "a periodic check of the air conditioner",
    ),
    "food vendor": (
        "a dinner order for six people",
        "lunch boxes for a neighborhood meeting",
        "a weekly vegetarian meal delivery",
        "snacks for a small family gathering",
        "a birthday cake with a custom message",
        "an evening tea and tiffin order",
        "a catered lunch for a community event",
        "a breakfast order for visiting relatives",
    ),
}

OPENERS = {
    "positive": (
        "I booked a local {provider} to handle {task}.",
        "A nearby {provider} came out to take care of {task}.",
        "When I needed {task}, I contacted a {provider} from the neighborhood.",
        "The booking was for {task}, and a local {provider} took it on.",
        "I arranged {task} with a {provider} after checking the local listing.",
        "We asked a nearby {provider} to help with {task}.",
        "My request for {task} was handled by a local {provider}.",
        "I reached out to a {provider} about {task}.",
        "The {provider} I contacted agreed to take care of {task}.",
        "I scheduled a neighborhood {provider} for {task}.",
        "For {task}, I chose a local {provider} rather than a chain.",
        "Our appointment with a {provider} covered {task}.",
    ),
    "neutral": (
        "I booked a local {provider} to handle {task}.",
        "The service request covered {task}, with a nearby {provider} assigned.",
        "I contacted a {provider} about {task} and received an appointment slot.",
        "A local {provider} was scheduled to take care of {task}.",
        "My booking listed {task} as the requested service.",
        "The appointment was with a {provider} for {task}.",
        "I submitted a request for {task} through a neighborhood {provider}.",
        "A {provider} accepted the request to handle {task}.",
        "The listing described a {provider} who could take on {task}.",
        "I set up a visit with a nearby {provider} about {task}.",
        "The scheduled service involved {task} at my home.",
        "A local {provider} received my request for {task}.",
    ),
    "negative": (
        "I hired a nearby {provider} to handle {task}.",
        "I contacted a {provider} after realizing I needed {task}.",
        "The booking was for {task}, arranged with a local {provider}.",
        "A neighborhood {provider} accepted my request for {task}.",
        "I scheduled a local {provider} to take care of {task}.",
        "My appointment with a {provider} was meant to cover {task}.",
        "I asked a nearby {provider} to deal with {task}.",
        "The request for {task} went to a local {provider}.",
        "I booked the {provider} after asking for help with {task}.",
        "A {provider} from the area agreed to handle {task}.",
        "The service I arranged with a {provider} was for {task}.",
        "I reached out to a neighborhood {provider} to sort out {task}.",
    ),
}

FOLLOW_UPS = {
    "plumber": {
        "positive": (
            "The leak stopped, water pressure is steady, and the old fitting was taken away.",
            "Arrival was on time, the quote stayed fixed, and the repair works well.",
            "They explained the cause, protected the counter, and left the area clean.",
            "Messages were answered quickly, and the job was finished in one visit.",
            "The new fitting works smoothly, and the workmanship looks careful.",
            "Costs were explained in advance, and the final bill matched the quote.",
        ),
        "neutral": (
            "The plumber measured the pipe and listed two replacement fittings.",
            "Water pressure was checked before and after the appointment.",
            "The visit lasted about an hour, and a written estimate was provided.",
            "The plumber arrived at the booked time and recorded the tap model.",
            "Payment was made after the listed plumbing task was completed.",
            "The appointment notes included the pipe size and the work requested.",
        ),
        "negative": (
            "The tap is still leaking, and my request for a return visit got no reply.",
            "The plumber arrived late, left debris under the sink, and charged extra.",
            "Water pressure is unchanged, although the invoice lists the repair as finished.",
            "They replaced the wrong fitting and would not correct the mistake.",
            "The quoted amount increased after the work, without prior explanation.",
            "I sent photos of the continuing leak, but the provider stopped responding.",
        ),
    },
    "electrician": {
        "positive": (
            "The switch now works reliably, and the electrician tested it before leaving.",
            "They explained the safety issue in plain language and charged the quoted price.",
            "The fans were fitted neatly, the wiring was checked, and the room was tidy.",
            "The electrician arrived as promised and answered every question patiently.",
            "The work was careful and professional, with a clear breakdown of the cost.",
            "A quick response and a thorough safety check made this excellent value.",
        ),
        "neutral": (
            "The electrician tested the circuit and noted the switch rating.",
            "The appointment began at the scheduled time and lasted 45 minutes.",
            "A parts list and a written estimate were provided after inspection.",
            "The work order records the number of sockets that were checked.",
            "The electrician called before arrival to confirm access to the panel.",
            "The service receipt lists the inspection and the materials used.",
        ),
        "negative": (
            "The outlet is still unsafe, and the electrician left before testing it.",
            "A new charge for materials appeared after the quote had been accepted.",
            "The fan wobbles, exposed wiring was left behind, and calls went unanswered.",
            "They missed the appointment window and sent no message about the delay.",
            "The repair failed the next day, but the provider refused to return.",
            "The electrician rushed the job and gave no clear explanation of the bill.",
        ),
    },
    "cleaner": {
        "positive": (
            "The kitchen and bathrooms were spotless, and the cleaner arrived on time.",
            "They checked the rooms with me and stayed within the agreed price.",
            "The floors were streak-free, the corners were thorough, and the flat felt fresh.",
            "Communication was easy, and the team finished before the guests arrived.",
            "They handled delicate surfaces carefully and left every room in good order.",
            "The service was detailed and professional, making the fee feel worthwhile.",
        ),
        "neutral": (
            "The cleaner covered the rooms listed in the booking form.",
            "The visit started at 9 a.m. and ended shortly before noon.",
            "The service receipt lists the kitchen, two bedrooms, and one bathroom.",
            "The cleaner brought the supplies noted in the appointment message.",
            "A checklist of completed rooms was shared at the end of the visit.",
            "The quoted amount was paid after the scheduled cleaning was finished.",
        ),
        "negative": (
            "Several surfaces were missed, and the cleaner left before the agreed time.",
            "The bathroom was still dirty, despite an extra fee for a deep clean.",
            "I found streaks on the windows and dust on the shelves after the visit.",
            "The team arrived late and did not respond when I asked about the delay.",
            "An unapproved add-on appeared on the final bill with no explanation.",
            "The work was rushed, and a second cleaner had to redo several rooms.",
        ),
    },
    "carpenter": {
        "positive": (
            "The cupboard door now closes properly, and the repair looks seamless.",
            "Measurements were checked twice, and the shelf fits exactly as requested.",
            "The carpenter protected the floor, cleaned up, and kept to the quote.",
            "They explained the wood options clearly and completed the work on schedule.",
            "The joinery is sturdy and neat, with careful attention to the finish.",
            "A courteous update before arrival made the whole repair straightforward.",
        ),
        "neutral": (
            "The carpenter measured the opening and wrote down the shelf dimensions.",
            "The visit covered one door and two cupboard hinges.",
            "The estimate listed the wood type, labor, and expected completion date.",
            "The carpenter arrived in the afternoon and inspected the chair joint.",
            "The receipt records the replacement handle and the labor charge.",
            "The requested measurements were sent by message after the appointment.",
        ),
        "negative": (
            "The shelf was cut to the wrong size, and the carpenter would not remake it.",
            "The cupboard still sticks, although the provider marked the repair complete.",
            "They scratched the floor, left wood shavings behind, and ignored my message.",
            "The final charge included timber that was never discussed in the quote.",
            "The carpenter cancelled twice and gave no new date for the unfinished work.",
            "The joint came loose again, and the provider declined to fix it.",
        ),
    },
    "appliance technician": {
        "positive": (
            "The washer drains properly now, and the technician explained the cause.",
            "They tested the refrigerator after replacing the part and kept to the estimate.",
            "The technician arrived promptly, worked carefully, and left the kitchen tidy.",
            "A clear explanation and a durable repair made the service good value.",
            "They answered my questions patiently and completed the repair in one visit.",
            "The appliance is running quietly again, and the price was fair.",
        ),
        "neutral": (
            "The technician recorded the appliance model and the serial number.",
            "The inspection took about 35 minutes, followed by a written estimate.",
            "The service note lists the part that was checked during the visit.",
            "The technician arrived during the appointment window and tested the controls.",
            "The receipt shows the inspection fee and the date of service.",
            "A follow-up date was entered on the service card before departure.",
        ),
        "negative": (
            "The washer still will not drain, and my follow-up request was ignored.",
            "The same refrigerator noise returned, but the technician would not come back.",
            "They charged for a part without showing it to me or explaining the price.",
            "The repair was rushed, and the appliance stopped working again that evening.",
            "The technician arrived late and left the machine disconnected.",
            "I received no update about the delay or the extra charge on the invoice.",
        ),
    },
    "food vendor": {
        "positive": (
            "Every meal was labeled correctly, arrived hot, and was ready at the agreed time.",
            "The portions were generous, the food tasted fresh, and the price was fair.",
            "The vendor confirmed the order promptly and handled our dietary notes carefully.",
            "The cake matched the requested design, and the pickup was right on time.",
            "The catering was well organized, flavorful, and excellent value for the group.",
            "Friendly communication and careful packaging made the delivery a success.",
        ),
        "neutral": (
            "The order contained six meals and came in labeled containers.",
            "The vendor confirmed the pickup time by message the previous evening.",
            "The receipt lists the meal count, delivery fee, and requested items.",
            "The food was delivered to the address in the booking details.",
            "The menu and final order matched the selection sent by email.",
            "The catering boxes were collected at the time shown on the receipt.",
        ),
        "negative": (
            "Several items were missing, and the vendor did not answer my message.",
            "The food arrived cold and late, despite the delivery time being confirmed.",
            "Two dietary requests were overlooked, and no replacement was offered.",
            "The final bill included a delivery fee that was never mentioned before ordering.",
            "The order was packed incorrectly, and the vendor stopped replying afterward.",
            "Our event started before the food arrived, with no update about the delay.",
        ),
    },
}


def read_existing_reviews() -> list[dict[str, str]]:
    with ORIGINAL_DATASET_PATH.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["review", "sentiment"]:
            raise ValueError("Dataset must have exactly review,sentiment columns")
        rows = []
        for row in reader:
            review = (row.get("review") or "").strip()
            sentiment = (row.get("sentiment") or "").strip().lower()
            if review and sentiment in LABELS:
                rows.append({"review": review, "sentiment": sentiment})
    return rows


def build_dataset() -> list[dict[str, str]]:
    rows = read_existing_reviews()
    randomizer = random.Random(RANDOM_SEED)
    seen_reviews = {row["review"].casefold() for row in rows}
    counts = Counter(row["sentiment"] for row in rows)

    if any(counts[label] > TARGET_PER_LABEL for label in LABELS):
        raise ValueError("Existing dataset has more than 500 examples for a label")

    for label in LABELS:
        target_additions = TARGET_PER_LABEL - counts[label]
        service_names = list(SERVICE_TASKS)
        randomizer.shuffle(service_names)
        service_quotas = {
            service: target_additions // len(service_names)
            for service in service_names
        }
        for service in service_names[: target_additions % len(service_names)]:
            service_quotas[service] += 1

        for service, tasks in SERVICE_TASKS.items():
            provider_tasks = [(service, task) for task in tasks]
            randomizer.shuffle(provider_tasks)
            candidates = []
            for provider, task in provider_tasks:
                for opener in OPENERS[label]:
                    for follow_up in FOLLOW_UPS[service][label]:
                        candidates.append(
                            f"{opener.format(provider=provider, task=task)} {follow_up}"
                        )
            randomizer.shuffle(candidates)
            added_for_service = 0
            for review in candidates:
                normalized_review = " ".join(review.split()).casefold()
                if normalized_review in seen_reviews:
                    continue
                seen_reviews.add(normalized_review)
                rows.append({"review": review, "sentiment": label})
                added_for_service += 1
                if added_for_service == service_quotas[service]:
                    break
            if added_for_service != service_quotas[service]:
                raise ValueError(f"Not enough unique reviews generated for {label}/{service}")

    randomizer.shuffle(rows)
    return rows


def main() -> None:
    rows = build_dataset()
    with DATASET_PATH.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=["review", "sentiment"])
        writer.writeheader()
        writer.writerows(rows)

    reviews = [row["review"] for row in rows]
    class_counts = Counter(row["sentiment"] for row in rows)
    print(f"Total samples: {len(rows)}")
    print(f"Class distribution: {dict(sorted(class_counts.items()))}")
    print(f"Duplicate count: {len(reviews) - len(set(reviews))}")
    print(f"Empty review count: {sum(not review.strip() for review in reviews)}")
    print(f"Minimum review length: {min(map(len, reviews))}")
    print(f"Maximum review length: {max(map(len, reviews))}")


if __name__ == "__main__":
    main()