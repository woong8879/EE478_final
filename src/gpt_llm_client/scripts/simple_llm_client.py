#!/usr/bin/env python3
# Patched for openai >= 1.0: the old `openai.ChatCompletion` API was removed.
# Falls back to the legacy call only if the new SDK is not installed.
import rospy
import os
from gpt_llm_client.srv import LLMQuery, LLMQueryResponse


class StatelessLLMClient:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            rospy.logerr("OPENAI_API_KEY not set")
            raise RuntimeError("Missing OPENAI_API_KEY")

        ################################### DO NOT USE BIGGER MODEL ###################################
        self.model = rospy.get_param("~model", "gpt-4.1-nano") ### Please use "gpt-4o-mini" or "gpt-4.1-nano" only. DO NOT USE BIGGER MODEL
        ################################### DO NOT USE BIGGER MODEL ###################################

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
            self._mode = "v1"
        except ImportError:
            import openai
            openai.api_key = api_key
            self._client = openai
            self._mode = "v0"

        self.service = rospy.Service("llm_query", LLMQuery, self.handle)
        rospy.loginfo(f"Stateless LLM client ready (openai mode={self._mode}, model={self.model})")

    def handle(self, req):
        try:
            messages = [
                {"role": "system", "content": "You are a robot assistant."},
                {"role": "user",   "content": req.prompt},
            ]
            if self._mode == "v1":
                resp = self._client.chat.completions.create(
                    model=self.model, messages=messages,
                )
                answer = resp.choices[0].message.content
            else:
                resp = self._client.ChatCompletion.create(
                    model=self.model, messages=messages,
                )
                answer = resp["choices"][0]["message"]["content"]
            return LLMQueryResponse(answer)

        except Exception as e:
            rospy.logerr(f"OpenAI error: {e}")
            return LLMQueryResponse(f"ERROR: {e}")


if __name__ == "__main__":
    rospy.init_node("stateless_llm_client")
    node = StatelessLLMClient()
    rospy.spin()
