"""
tests/test_intent_classifier.py
===============================
Unit tests for core/intent_classifier.py to verify conservative routing
between pure conversational turns (fast path) and action/tool requests (planner path).
"""
import unittest
from core.intent_classifier import is_pure_conversational


class TestIntentClassifier(unittest.TestCase):

    def test_pure_conversational_greetings(self):
        self.assertTrue(is_pure_conversational("hello"))
        self.assertTrue(is_pure_conversational("hi"))
        self.assertTrue(is_pure_conversational("hey jarvis"))
        self.assertTrue(is_pure_conversational("good morning"))
        self.assertTrue(is_pure_conversational("good evening!"))

    def test_pure_conversational_smalltalk(self):
        self.assertTrue(is_pure_conversational("how are you?"))
        self.assertTrue(is_pure_conversational("who are you?"))
        self.assertTrue(is_pure_conversational("what can you do?"))
        self.assertTrue(is_pure_conversational("tell me a joke"))
        self.assertTrue(is_pure_conversational("thank you so much"))

    def test_action_routing_for_all_5_failure_cases(self):
        # Case a: Weather
        self.assertFalse(is_pure_conversational("what's the weather?"))
        # Case b: Weather in city
        self.assertFalse(is_pure_conversational("what's the weather in Tokyo?"))
        # Case c: App opening
        self.assertFalse(is_pure_conversational("open WhatsApp"))
        # Case d: File deletion
        self.assertFalse(is_pure_conversational("delete freshers folder on desktop"))
        # Case e: Code generation
        self.assertFalse(is_pure_conversational("write a python script that reverses a string and save it"))

    def test_action_routing_for_os_and_system_commands(self):
        self.assertFalse(is_pure_conversational("set volume to 50"))
        self.assertFalse(is_pure_conversational("turn off wifi"))
        self.assertFalse(is_pure_conversational("take a screenshot"))
        self.assertFalse(is_pure_conversational("shut down"))
        self.assertFalse(is_pure_conversational("remind me to call John at 5pm"))
        self.assertFalse(is_pure_conversational("search for quantum computing news"))
        self.assertFalse(is_pure_conversational("send a message to Mom saying hello"))
        self.assertFalse(is_pure_conversational("remember that my favorite color is blue"))


if __name__ == "__main__":
    unittest.main()
