from collections import Counter, defaultdict
from datetime import date
from operator import itemgetter
from typing import Any, Optional

from matplotlib.axes import Axes
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from dateutil.relativedelta import relativedelta

from patient_tpp.basics import is_outcome_of_interest

# Most patient data has a left bounds of 2005-Q4.
_default_start = date(2005, 10, 1)

DAYS_IN_YEAR = 365.25


def construct_dates(deltats: list[float], start: date = _default_start) -> list[date]:
    """
    Construct a list of date objects corresponding to the passed delta.
    """
    dates = [start]
    for t in deltats:
        dates.append(dates[-1] + relativedelta(days=int(t * DAYS_IN_YEAR / 4)))
    return dates[1:]


def dither_date(q: date, scale: int = 20) -> date:
    return q + relativedelta(days=np.random.randint(-scale, scale))


def dither(q: float, scale: float = 0.05) -> float:
    a = q * scale
    return q + np.random.ranf() * a - a


def patient_timeline(
    s: dict[str, Any],
    prediction: tuple[list[float], list[Any]],
    ax: Axes,
    enc2thingname: dict[int, str],
    lookback: int = 6,
    window: tuple[Optional[date], Optional[date]] = (None, None),
) -> Axes:
    the_dates = construct_dates(s["time_since_last_event"])[:-1]
    lookback = min(lookback, len(s["type_event"]))

    ax.set(
        title=f"ptid {s['ptid']} timeline, \
{the_dates[0].strftime('%b %Y')} through {the_dates[-1].strftime('%b %Y')}"
    )

    # The vertical stems.
    levels = [
        (i % 8) + 1 if i % 2 else -((i % 8) + 1) for i in range(len(s["type_event"]))
    ]
    ax.vlines(
        the_dates,
        0,
        levels,
        color=[
            ("tab:red", 1 if is_outcome_of_interest(enc2thingname[c]) else 0.5)
            for c in s["type_event"]
        ],
    )
    # The baseline.
    ax.axhline(0, c="black")
    # The markers on the baseline.
    ind_dates = [
        d
        for d, cond in zip(the_dates, s["type_event"])
        if not is_outcome_of_interest(enc2thingname[cond])
    ]
    oi_dates = [
        d
        for d, cond in zip(the_dates, s["type_event"])
        if is_outcome_of_interest(enc2thingname[cond])
    ]
    ax.plot(ind_dates, np.zeros_like(ind_dates), "ko", mfc="white")
    ax.plot(oi_dates, np.zeros_like(oi_dates), "ko", mfc="tab:red")

    # Annotate the lines.
    for the_date, level, cond in zip(the_dates, levels, s["type_event"]):
        thing = enc2thingname[cond]
        ax.annotate(
            thing,
            xy=(dither_date(the_date), dither(level)),
            xytext=(-3, np.sign(level) * 3),
            textcoords="offset points",
            verticalalignment="bottom" if level > 0 else "top",
            weight="bold" if is_outcome_of_interest(thing) else "normal",
            bbox={"boxstyle": "square", "pad": 0, "lw": 0, "fc": (1, 1, 1, 0.7)},
        )

    if prediction is not None:
        pred_dtimes, candidates = prediction
        the_pred_dates = construct_dates(pred_dtimes, start=the_dates[-lookback])
        pred_levels = [
            (i % 8) + 1 if i % 2 else -((i % 8) + 1)
            for i in range(
                len(s["type_event"]),
                len(s["type_event"]) + len([x for y in candidates for x in y]),
            )
        ]
        thing_labels = []
        scores = []
        oi_dates = []
        for pos, top_n in enumerate(candidates):
            if len(top_n) == 0:
                continue
            for condition, score in top_n:
                thing_labels.append(f"{condition} ({score:.2f})*")
                scores.append(score)
                # print(pos, the_pred_dates, candidates)
                oi_dates.append(the_pred_dates[pos])
        ax.vlines(oi_dates, 0, pred_levels, color=["tab:green" for _ in scores])
        ax.scatter(
            oi_dates,
            np.zeros_like(oi_dates),
            s=np.array(scores) * 30,
            color="tab:green",
        )
        for the_date, level, thing_label in zip(oi_dates, pred_levels, thing_labels):
            ax.annotate(
                thing_label,
                xy=(the_date, level),
                xytext=(-3, np.sign(level) * 3),
                textcoords="offset points",
                verticalalignment="bottom" if level > 0 else "top",
                weight="bold",  # if is_outcome_of_interest(thing) else "normal",
                bbox={"boxstyle": "square", "pad": 0, "lw": 0, "fc": (1, 1, 1, 0.7)},
            )

    ax.xaxis.set(
        major_locator=mdates.YearLocator(), major_formatter=mdates.DateFormatter("%Y")
    )

    # Shade backtesting region
    ax.fill_between(
        the_dates,
        -15,
        15,
        where=([d >= the_dates[-lookback] for d in the_dates]),
        color="lightgreen",
        alpha=0.15,
    )

    # Remove the y-axis and some spines.
    ax.yaxis.set_visible(False)
    ax.spines[["left", "top", "right"]].set_visible(False)

    ax.margins(y=0.1)
    ax.set_ylim([-10, 10])
    ax.set_xlim(*window)
    return ax


def indexed_patient_timeline(
    s_before: dict[str, Any],
    s_after: dict[str, Any],
    indexed_date: date,
    codes_oi: set[int],
    ax: Axes,
    enc2thingname: dict[int, str],
    window: tuple[Optional[date], Optional[date]],
) -> Axes:
    """
    Variation with a "before" and "after" demarcated by an indexed date.
    """
    the_dates_b = construct_dates(s_before["time_since_last_event"])[:-1]
    the_dates_a = construct_dates(s_after["time_since_last_event"])[:-1]

    ax.set(
        title=f"ptid {s_before['ptid']} timeline, {the_dates_b[0].strftime('%b %Y')} \
through {the_dates_a[-1].strftime('%b %Y')}"
    )

    # The vertical stems.
    levels_b = [
        (i % 8) + 1 if i % 2 else -((i % 8) + 1)
        for i in range(len(s_before["type_event"]))
    ]
    ax.vlines(
        the_dates_b,
        0,
        levels_b,
        color=[
            ("tab:red", 1 if c in codes_oi else 0.5) for c in s_before["type_event"]
        ],
    )
    # The baseline.
    ax.axhline(0, c="black")
    # The markers on the baseline.
    ind_dates = [
        d
        for d, cond in zip(the_dates_b, s_before["type_event"])
        if cond not in codes_oi
    ]
    oi_dates = [
        d for d, cond in zip(the_dates_b, s_before["type_event"]) if cond in codes_oi
    ]
    ax.plot(ind_dates, np.zeros_like(ind_dates), "ko", mfc="white")
    ax.plot(oi_dates, np.zeros_like(oi_dates), "ko", mfc="tab:red")

    levels_a = [
        (i % 8) + 1 if i % 2 else -((i % 8) + 1)
        for i in range(len(s_after["type_event"]))
    ]
    ax.vlines(
        the_dates_a,
        0,
        levels_a,
        color=[("tab:red", 1 if c in codes_oi else 0.5) for c in s_after["type_event"]],
    )
    # The baseline.
    ax.axhline(0, c="black")
    # The markers on the baseline.
    ind_dates = [
        d for d, cond in zip(the_dates_a, s_after["type_event"]) if cond not in codes_oi
    ]
    oi_dates = [
        d for d, cond in zip(the_dates_a, s_after["type_event"]) if cond in codes_oi
    ]
    ax.plot(ind_dates, np.zeros_like(ind_dates), "ko", mfc="white")
    ax.plot(oi_dates, np.zeros_like(oi_dates), "ko", mfc="tab:red")

    # Annotate the lines.
    for the_date, level, cond in zip(the_dates_b, levels_b, s_before["type_event"]):
        thing = enc2thingname[cond]
        ax.annotate(
            thing,
            xy=(dither_date(the_date), dither(level)),
            xytext=(-3, np.sign(level) * 3),
            textcoords="offset points",
            verticalalignment="bottom" if level > 0 else "top",
            weight="bold" if is_outcome_of_interest(thing) else "normal",
            bbox={"boxstyle": "square", "pad": 0, "lw": 0, "fc": (1, 1, 1, 0.7)},
        )
    for the_date, level, cond in zip(the_dates_a, levels_a, s_after["type_event"]):
        thing = enc2thingname[cond]
        ax.annotate(
            thing,
            xy=(dither_date(the_date), dither(level)),
            xytext=(-3, np.sign(level) * 3),
            textcoords="offset points",
            verticalalignment="bottom" if level > 0 else "top",
            weight="bold" if is_outcome_of_interest(thing) else "normal",
            bbox={"boxstyle": "square", "pad": 0, "lw": 0, "fc": (1, 1, 1, 0.7)},
        )

    ax.xaxis.set(
        major_locator=mdates.YearLocator(), major_formatter=mdates.DateFormatter("%Y")
    )

    # Shade backtesting region
    ax.fill_between(
        the_dates_a,
        -15,
        15,
        where=([d >= indexed_date for d in the_dates_a]),
        color="lightgreen",
        alpha=0.15,
    )

    # Remove the y-axis and some spines.
    ax.yaxis.set_visible(False)
    ax.spines[["left", "top", "right"]].set_visible(False)

    ax.margins(y=0.1)
    ax.set_ylim([-10, 10])
    ax.set_xlim(*window)
    return ax


def count_up(marks: list[int], enc2thingname: dict[int, str]) -> list[tuple[str, int]]:
    c = Counter(marks)
    outdict: dict[str, int] = defaultdict(int)
    for k, v in c.items():
        try:
            thingname = enc2thingname[k]
        except KeyError:
            continue
        outdict[thingname] += v
    return sorted(outdict.items())


def otherize(tally: dict[Any, int], n: int = 8) -> list[tuple[Any, int]]:
    flat_tally = sorted(tally.items(), key=itemgetter(1))[::-1]
    otherized = flat_tally[:n] + [(("Other", "*"), sum([x[1] for x in flat_tally[n:]]))]
    return otherized


def patient_piechart(
    s: dict[str, Any], ax: Axes, enc2thingname: dict[int, str]
) -> Axes:
    tally = count_up(s["type_event"], enc2thingname)
    plt.pie([x[1] for x in tally], labels=[x[0] for x in tally], autopct="%.1f")
    ax.set_title(
        f"Patient {s['ptid']} condition breakdown\n{sum([x[1] for x in tally])} conditions recorded"
    )
    return ax


def coi_only_in_lookback(
    s: dict[str, Any], enc2thingname: dict[int, str], lookback: int = 6
) -> bool:
    is_only_in_lookback = False
    if (
        len(s["type_event"]) > lookback
        and all(
            [
                not is_outcome_of_interest(enc2thingname[c])
                for c in s["type_event"][:-lookback]
            ]
        )
        and any(
            [
                is_outcome_of_interest(enc2thingname[c])
                for c in s["type_event"][-lookback:]
            ]
        )
    ):
        is_only_in_lookback = True
    return is_only_in_lookback


def contains_coi_not_in_prelookback(
    s: dict[str, Any], enc2thingname: dict[int, str], lookback: int = 6
) -> bool:
    keep = True
    if len(s["type_event"]) <= lookback:
        keep = False
    else:
        for c in s["type_event"][:-lookback]:
            if (
                is_outcome_of_interest(enc2thingname[c])
                and c in s["type_event"][-lookback:]
            ):
                keep = False
                break
    return keep


def rmse(a: list[float], b: list[float]) -> float:
    return np.sqrt(np.mean((np.array(a) - np.array(b)) ** 2))
