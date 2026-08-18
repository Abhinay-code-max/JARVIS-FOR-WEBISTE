"""
tests/test_tts.py
=================
Unit tests for core/tts.py, verifying emoji filtering and TTS player behavior.
"""
import unittest
from unittest.mock import MagicMock

from core.tts import _strip_emojis, TTSPlayer


class EmojiStrippingTest(unittest.TestCase):
    def test_strips_common_emojis(self):
        self.assertEqual(_strip_emojis("Hello! 😊 How are you? 👍"), "Hello! How are you?")

    def test_strips_weather_and_symbol_emojis(self):
        self.assertEqual(_strip_emojis("The weather is sunny ☀️ with 28°C."), "The weather is sunny with 28°C.")
        self.assertEqual(_strip_emojis("Flight booked ✈️ for tomorrow 🚀"), "Flight booked for tomorrow")

    def test_strips_consecutive_and_compound_emojis(self):
        self.assertEqual(_strip_emojis("Awesome! 🎉🥳✨"), "Awesome!")

    def test_preserves_standard_text_and_punctuation(self):
        txt = "Good morning, sir. All 5 tasks completed successfully (status: 200)."
        self.assertEqual(_strip_emojis(txt), txt)

    def test_empty_or_whitespace_handling(self):
        self.assertEqual(_strip_emojis(""), "")
        self.assertEqual(_strip_emojis(None), "")
        self.assertEqual(_strip_emojis("   😊   "), "")


class TTSPlayerEmojiIntegrationTest(unittest.TestCase):
    def test_player_speaks_cleaned_text(self):
        mock_engine = MagicMock()
        player = TTSPlayer(mock_engine)

        player.speak("Hello sir! 😊 Welcome back. 👍")
        mock_engine.speak.assert_called_once_with("Hello sir! Welcome back.")

    def test_player_skips_pure_emoji_text(self):
        mock_engine = MagicMock()
        player = TTSPlayer(mock_engine)

        player.speak("😊👍🎉")
        mock_engine.speak.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
