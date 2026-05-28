#!/usr/bin/env python3
# Patched for openai >= 1.0: the old `openai.ChatCompletion` API was removed.
import rospy
import os
import json
import time
from gpt_llm_client.srv import LLMQuery, LLMQueryResponse

MAX_HISTORY = 20
SUMMARY_TRIGGER = 10


def _make_client(api_key):
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key), "v1"
    except ImportError:
        import openai
        openai.api_key = api_key
        return openai, "v0"


def _chat(client, mode, model, messages):
    if mode == "v1":
        r = client.chat.completions.create(model=model, messages=messages)
        return r.choices[0].message.content
    r = client.ChatCompletion.create(model=model, messages=messages)
    return r["choices"][0]["message"]["content"]


class MemoryLLMClient:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            rospy.logerr("OPENAI_API_KEY not set")
            raise RuntimeError("Missing OPENAI_API_KEY")

        self._client, self._mode = _make_client(api_key)

        ################################### DO NOT USE BIGGER MODEL ###################################
        self.model = rospy.get_param("~model", "gpt-4.1-nano") ### Please use "gpt-4o-mini" or "gpt-4.1-nano" only. DO NOT USE BIGGER MODEL
        ################################### DO NOT USE BIGGER MODEL ###################################

        # history directory
        pkg_path = os.path.dirname(os.path.abspath(__file__))
        self.history_dir = os.path.join(pkg_path, "history")
        os.makedirs(self.history_dir, exist_ok=True)

        timestamp = int(time.time())
        self.history_file = os.path.join(
            self.history_dir,
            f"session_{timestamp}.jsonl"
        )

        self.turn_history = []
        self.summary_memory = ""

        summary_path = os.path.join(self.history_dir, "summary.json")
        self.summary_path = summary_path
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                self.summary_memory = json.load(f).get("summary", "")

        self.service = rospy.Service("llm_query", LLMQuery, self.handle)
        rospy.loginfo("Memory-aware LLM client ready")

    def save_entry(self, role, content):
        entry = {"role": role, "content": content}
        self.turn_history.append(entry)

        if len(self.turn_history) > MAX_HISTORY:
            self.turn_history = self.turn_history[-MAX_HISTORY:]

        with open(self.history_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def build_messages(self, user):
        msgs = [
            {"role": "system",
             "content": "You are a robot assistant with long-term memory."}
        ]
        if self.summary_memory:
            msgs.append({"role": "system",
                         "content": f"[SUMMARY]\n{self.summary_memory}"})

        msgs.extend(self.turn_history)
        msgs.append({"role": "user", "content": user})
        return msgs

    def update_summary(self):
        if len(self.turn_history) < SUMMARY_TRIGGER:
            return

        try:
            new_summary = _chat(self._client, self._mode, self.model, [
                {"role": "system", "content": "Summarize this memory."},
                {"role": "user", "content": json.dumps(self.turn_history)}
            ]).strip()

            self.summary_memory = new_summary
            with open(self.summary_path, "w") as f:
                json.dump({"summary": new_summary}, f, indent=2)

            self.turn_history = []
            rospy.loginfo("Updated memory summary")

        except Exception as e:
            rospy.logerr(f"Summary update failed: {e}")

    def handle(self, req):
        try:
            msgs = self.build_messages(req.prompt)
            answer = _chat(self._client, self._mode, self.model, msgs)

            self.save_entry("user", req.prompt)
            self.save_entry("assistant", answer)
            self.update_summary()

            return LLMQueryResponse(answer)

        except Exception as e:
            rospy.logerr(f"OpenAI error: {e}")
            return LLMQueryResponse(f"ERROR: {e}")


if __name__ == "__main__":
    rospy.init_node("memory_llm_client")
    node = MemoryLLMClient()
    rospy.spin()
