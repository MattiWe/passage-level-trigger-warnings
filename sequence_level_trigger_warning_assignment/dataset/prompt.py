import json
import logging
from pathlib import Path

class Prompt:
    def __init__(self, persona: bool = False, definition: bool = False, extended_instruction: bool = False,
                 demonstration: bool = False, explanation: bool = False,
                 num_unanimous: int = 1, num_non_unanimous: int = 0, 
                 prompt_resource_dir: Path = Path("../../../resources/prompt")):
        """
        :param persona:
        :param definition:
        :param extended_instruction: Add the prompt part `extended-instruction`
        :param demonstration:
        :param explanation:
        :param num_unanimous: how many demonstrations to apppend (positive and negative, so `1` would select 2 demonstraions)
        :param num_non_unanimous:
        """
        self.has_persona = persona
        self.has_definition = definition
        self.has_demonstration = demonstration
        self.has_explanation = explanation
        self.num_unanimous = num_unanimous
        self.num_non_unanimous = num_non_unanimous
        self.has_extended_instruction = extended_instruction
        self.prompt_parts = json.loads(open(prompt_resource_dir / "gpt-prompt-parts.json").read())
        self.demonstrations_unanimous = json.loads(open(prompt_resource_dir / "gpt-prompt-unanimous-demonstrations.json").read())
        self.demonstrations_non_unanimous = json.loads(open(prompt_resource_dir / "gpt-prompt-non-unanimous-demonstrations.json").read())
        self.demonstrations_selected = json.loads(open(prompt_resource_dir / "gpt-prompt-selected-demonstrations.json").read())

    def __str__(self) -> str:
        return f"{'extended-' if self.has_extended_instruction else ''}instruction" \
               f"{'-persona' if self.has_persona else ''}" \
               f"{'-definition' if self.has_definition else ''}" \
               f"{'-explanation' if self.has_explanation else ''}" \
               f"{'-' + str(self.num_unanimous) + '-unanimous' if self.has_demonstration else ''}" \
               f"{'-' + str(self.num_non_unanimous) + '-non-unanimous' if self.has_demonstration else ''}"

    def __call__(self, text: str, warning: str) -> str:
        """
        Construct the query from
        :param text:
        :param warning:
        :return:
        """
        query = ""
        if self.has_persona:
            query += f"{self.prompt_parts[warning]['persona']} "
        if self.has_definition:
            query += f"{self.prompt_parts[warning]['definition'] if self.has_definition else ''}"
            query += f"{self.prompt_parts['extended-instruction'] if self.has_definition else ''}\n"

        if self.has_demonstration:
            query += f"Consider the following examples:\n"
            for idx in range(0, max(self.num_non_unanimous, self.num_unanimous)):
                if idx < self.num_unanimous:
                    query += f"{self.prompt_parts['text_prefix']} {self.demonstrations_unanimous[warning]['positive'][idx]}\n" \
                             f"{self.prompt_parts['tw_prompt']} {self.prompt_parts['yes']}\n"
                    query += f"{self.prompt_parts['text_prefix']} {self.demonstrations_unanimous[warning]['negative'][idx]}\n" \
                             f"{self.prompt_parts['tw_prompt']} {self.prompt_parts['no']}n"

                if idx < self.num_non_unanimous:
                    query += f"{self.prompt_parts['text_prefix']} {self.demonstrations_non_unanimous[warning]['positive'][idx]}\n" \
                             f"{self.prompt_parts['tw_prompt']} {self.prompt_parts['yes']}\n"
                    query += f"{self.prompt_parts['text_prefix']} {self.demonstrations_non_unanimous[warning]['negative'][idx]}\n" \
                             f"{self.prompt_parts['tw_prompt']} {self.prompt_parts['no']}\n"
                    
        query += f"{self.prompt_parts[warning]['instruction']}\n "
        query += f"{self.prompt_parts['text_prefix']} {text}\n {self.prompt_parts['tw_prompt']} "
        return query

    @staticmethod
    def parse(text: str):
        text = text.lower()
        if not (text.startswith("yes") or
                text.startswith("warning: yes") or
                text.startswith("warning: no") or
                text.startswith("no")):
            logging.info(f'invalid response: {text}')
        if 'yes' in text:
            return 1
        elif 'no' in text:
            return 0
        elif text == '':
            # Here the generation was probably blocked for safety reasons
            logging.warning('empty text')
            return 1
        else:
            logging.error(f"error parsing response {text}")
            return 0