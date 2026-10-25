"""Stubs the GUI suites share (they run offscreen: every dialog is answered, never shown).

stub_track_choice: (G217) the Track pop-up - asked when a press has data ahead - answers `answer` ("keep",
"fresh", "gaps", or None = cancelled) and records what it would have said and the counts it was given. Every suite that presses Track installs it
right after importing the app, so a press with data ahead never waits for a click.
"""


class TrackChoiceStub:
    def __init__(self, answer):
        self.answer = answer
        self.said: list[str] = []          # the pop-up's sentence of each press that asked
        self.infos: list[dict] = []        # ... and the counts it was given


def stub_track_choice(answer="keep") -> TrackChoiceStub:
    from kinetrace.app import MainWindow
    from kinetrace.trackchoice import ahead_sentence
    st = TrackChoiceStub(answer)

    def ask(self, info):
        st.said.append(ahead_sentence(info))
        st.infos.append(dict(info))
        return st.answer
    MainWindow._ask_track_choice = ask
    return st
