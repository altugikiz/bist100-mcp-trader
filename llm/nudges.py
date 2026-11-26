"""Final-answer instructions the agent loop adds once the tools have returned (one per question type).

The same texts are put into the fine-tuning transcripts (llm/dataset_research.py), so the model has seen exactly the
conversation format it meets at inference. They steer the wording only: every number must come from the tool results.
"""

SUFFIX = " Mention only companies and tickers that appear in the question or in the tool results; write tickers exactly as there."

_TEXT = {
    "macro": ("The tool results above already answer the question. Write the final answer now, in the question's language, about the "
              "stock and the macro variable named in the question only: the period and number of observations, the correlation r with its "
              "p-value, the first-half and second-half correlation (is the sign the same), the currency change and the stock's return in TRY "
              "and in currency terms, and the tool warnings. Say it is an association, not proof that the variable caused the moves. "
              "Use only numbers from the tool results and end with the label INVESTIGATE."),
    "events": ("The tool results above already answer the question. Write the final answer now, in the question's language, "
               "quoting the summary_lines numbers exactly and computing no new percentages or differences yourself: total events, how many rose within 5 days and how many did not, "
               "the average forward returns at 1, 3, 5 and 10 days, the two comparisons (RSI trough + high volume; bearish "
               "Supertrend + below the Ichimoku cloud), any small-sample warning, and the statement about events excluded "
               "because their forward window is incomplete."),
    "state": ("The tool result above already answers the question. Write the final answer now, in the question's language, "
              "quoting evidence_summary exactly: the rebound rate within 5 days for the matching events, the all-days "
              "baseline, the sample size n, and whether it beats the baseline; say that events overlap and the sample is "
              "a research indication only."),
    "weekday": ("The tool result above already answers the question. Write the final answer now, in the question's language, "
                "quoting evidence_summary exactly and computing no new numbers yourself: how often the full pattern occurred, the in-sample and out-of-sample "
                "probability against the unconditional probability with their sample sizes, the mean and median hold return "
                "and the return after costs, the multiple-testing risk, and say clearly whether the pattern is stable."),
    "fund": ("The tool results above already answer the question. Write the final answer now, in the question's language, "
             "without computing any new numbers: list every line of quarterly_history with its estimated disclosure date, EBITDA margin, "
             "net debt/TTM EBITDA and abnormal returns; quote improvement_vs_reaction; state that disclosure dates are estimates "
             "(quarter end plus 30 or 45 days), not real KAP timestamps; say what the 'vs sector' benchmark is (benchmark_type; "
             "XU100 when the stock has no sector peers); say it is association, not causation, with small samples; and say the "
             "20-day window after the latest quarter is incomplete if the tool says so."),
    "catchup": ("The tool results above already answer the question. Write the final answer now, in the question's language: "
                "name the most lagging stock with its lag_5d_pct, lag_10d_pct and lag_20d_pct from laggard_diagnosis (percentages as given, no recomputing), then quote the backtest evidence_summary for "
                "5, 10 and 20 days (catch-up rate, baseline, p-value, false positives), the label verdict, and the "
                "laggard_diagnosis (structural or temporary, with its EBITDA margin and macro numbers). Say the label "
                "POTENTIAL_CATCH_UP_CANDIDATE only if the label verdict supports it."),
}


def final_instruction(kind: str, advice_ticker: str | None = None) -> str:
    """kind: macro, events, state, weekday, fund, catchup, or advice (advice_ticker = the stock named in the question, if any)."""
    if kind == "advice":
        text = ("Write the final answer now, in the question's language. You cannot give a buy or sell call, a guarantee or a price target, "
                "and a claim that a stock will surely rise is not verified by data: say so plainly. "
                + ("Then quote the relationship found by analyze_macro_stock_relationship for " + advice_ticker + " exactly as the tool gives it "
                   "(period, number of observations, correlation r and p-value, first-half and second-half r, currency change, stock return in TRY "
                   "and in currency terms) and its warnings; correlation is not causation and a past relationship is not a forecast. "
                   if advice_ticker else "No stock was analysed, so offer to test a named stock instead. ")
                + "Use only numbers from the tool results. End with the label INVESTIGATE.")
    else:
        text = _TEXT[kind]
    return text + SUFFIX
