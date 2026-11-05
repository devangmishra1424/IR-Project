import pytest

from maillens.prep.cleaning import clean_body, normalize_subject, split_body


# ---------------------------------------------------------------------------- subjects
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("RE: Gas   Prices", "gas prices"),
        ("Fw: RE: re: Meeting Friday", "meeting friday"),
        ("FWD:Budget", "budget"),
        ("Regarding the budget", "regarding the budget"),  # "Re" inside a word is untouched
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_subject(raw, expected):
    assert normalize_subject(raw) == expected


# ---------------------------------------------------------------------------- splitting
def test_original_message_block_is_quoted():
    body = "Sounds good, let's do it.\n\n-----Original Message-----\nFrom: Jeff\nSent: Monday\nCan we meet?"
    fresh, quoted = split_body(body)
    assert fresh.strip() == "Sounds good, let's do it."
    assert "Can we meet?" in quoted


def test_lotus_notes_forward_header_opens_the_quoted_part():
    body = (
        "FYI\n---------------------- Forwarded by Kay Mann/Corp/Enron on 03/09/2001 10:15 AM "
        "---------------------------\nThe article text."
    )
    fresh, quoted = split_body(body)
    assert fresh.strip() == "FYI"
    assert "The article text." in quoted


def test_outlook_header_block_without_a_divider_is_detected():
    body = "Thanks.\nFrom: Jeff Dasovich\nSent: Tuesday\nTo: Kay\nSubject: Contract\nBody here."
    fresh, quoted = split_body(body)
    assert fresh.strip() == "Thanks."
    assert quoted.startswith("From: Jeff Dasovich")


def test_a_from_line_in_ordinary_prose_is_not_a_quote_boundary():
    fresh, quoted = split_body("From: the desk of the CFO, a short note.\nNothing else follows.")
    assert quoted == ""
    assert "short note" in fresh


def test_one_line_name_on_date_header_opens_the_quoted_part():
    body = (
        "Hard to believe\n\nslafontaine@globalp.com on 03/05/2001 11:31:02 AM\n"
        "To: John.Arnold@enron.com\ncc:\nSubject: Re: ecommerce\n\njust saw your picture"
    )
    fresh, quoted = split_body(body)
    assert fresh.strip() == "Hard to believe"
    assert "just saw your picture" in quoted


def test_two_line_notes_header_moves_the_sender_name_into_the_quote():
    body = (
        "713-628-3104 - why?\n\n\nLuis Mena@ENRON\n11/13/2000 02:23 PM\n"
        "To: Eric Bass/HOU/ECT@ECT\ncc:\nSubject:\n\nWhat's your celly-cell number?"
    )
    cleaned = clean_body(body)
    assert cleaned.fresh == "713-628-3104 - why?"
    assert "Luis Mena" in cleaned.quoted and "celly-cell" in cleaned.quoted
    assert "To:" not in cleaned.quoted and "cc:" not in cleaned.quoted


def test_a_date_on_its_own_line_is_not_a_boundary_without_headers_below_it():
    fresh, quoted = split_body("See you then.\n03/05/2001 11:31 AM\nBring the slides.")
    assert quoted == ""
    assert "Bring the slides." in fresh


def test_hard_wrapped_forward_divider_leaves_no_residue():
    body = (
        "---------------------- Forwarded by Phillip K Allen/HOU/ECT on 02/11/2000 \n"
        "03:38 PM ---------------------------\n\n\nTim Heizenrader\n02/10/2000 12:55 PM\n"
        "To: James B Fallon/HOU/ECT@ECT\ncc: Tim Belden/HOU/ECT@ECT\n"
        "Subject: Western Strategy Briefing\n\nSlides for today's meeting are attached:"
    )
    cleaned = clean_body(body)
    assert cleaned.fresh == ""
    assert "PM ---" not in cleaned.quoted
    assert "Slides for today's meeting are attached:" in cleaned.quoted


def test_angle_bracket_email_addresses_are_not_mistaken_for_html():
    text = clean_body('Ask "Bass, Jason" <Jason.Bass2@compaq.com> about it').text
    assert "Jason Bass2" in text


def test_angle_bracket_lines_are_quoted_even_before_any_marker():
    fresh, quoted = split_body("My reply.\n> earlier text\nMore of my reply.")
    assert fresh == "My reply.\nMore of my reply."
    assert quoted == "> earlier text"


def test_on_date_wrote_line_opens_quoted_part():
    fresh, quoted = split_body(
        "Yes.\nOn Mon, Mar 5, 2001 at 9:00 AM, Jeff Skilling wrote:\nOld text"
    )
    assert fresh == "Yes."
    assert "Old text" in quoted


# ---------------------------------------------------------------------------- cleaning
def test_enron_legal_footer_is_removed():
    footer = (
        "\n**********************************************************************\n"
        "This e-mail is the property of Enron Corp. and/or its relevant affiliate and may "
        "contain confidential and privileged material for the sole use of the intended recipient.\n"
        "**********************************************************************\n"
    )
    cleaned = clean_body("Please review the draft." + footer)
    assert cleaned.text == "Please review the draft."


def test_unterminated_footer_does_not_swallow_the_rest_of_a_long_message():
    body = "Hello.\n**********\nThis e-mail is the property of Enron.\n" + "word " * 2000
    assert len(clean_body(body).text) > 5000  # footer span is bounded, not unlimited


def test_generic_disclaimer_paragraph_is_dropped_but_ordinary_confidential_mention_is_kept():
    disclaimer = (
        "This message is confidential and privileged. If you are not the intended recipient "
        "you must delete this message; unauthorized use is prohibited."
    )
    cleaned = clean_body(f"See attached.\n\n{disclaimer}")
    assert cleaned.text == "See attached."
    assert clean_body("The terms are confidential.").text == "The terms are confidential."


def test_urls_and_notes_routing_and_addresses_are_cleaned():
    body = "Mail Kay Mann/Corp/Enron@ENRON or kay.mann@enron.com, see http://example.com/x?y=1 now"
    text = clean_body(body).text
    assert "http" not in text and "@" not in text and "/Corp" not in text
    assert "Kay Mann" in text and "kay mann" in text


def test_html_and_entities_are_stripped():
    assert clean_body("<html><b>Hello</b>&nbsp;world &amp; co</html>").text == "Hello world & co"


def test_headers_inside_the_quoted_part_are_dropped_but_content_is_kept():
    body = (
        "Reply.\n-----Original Message-----\nFrom: A\nTo: B\nSubject: X\n\nThe real earlier text."
    )
    cleaned = clean_body(body)
    assert cleaned.fresh == "Reply."
    assert cleaned.quoted == "The real earlier text."


def test_separator_lines_and_long_ids_disappear():
    text = clean_body("Hello\n=========\nid 1075840601283 end\n_____").text
    assert "=" not in text and "1075840601283" not in text and "_" not in text


def test_forward_only_message_keeps_its_content_in_the_quoted_field():
    body = "---------------------- Forwarded by Tana Jones/HOU/ECT on 07/05/1999 ----\nThe agreement text."
    cleaned = clean_body(body)
    assert cleaned.fresh == ""
    assert "agreement text" in cleaned.quoted
    assert "agreement text" in cleaned.text


def test_length_is_bounded():
    cleaned = clean_body("word " * 100_000, max_chars=1000)
    assert len(cleaned.fresh) <= 1000 and len(cleaned.text) <= 1001


@pytest.mark.parametrize("empty", [None, ""])
def test_empty_bodies_are_fine(empty):
    cleaned = clean_body(empty)
    assert (cleaned.fresh, cleaned.quoted, cleaned.text) == ("", "", "")
