import re

from utils import load_text, jload





class BaselineProcessor:
    def __init__(self,
                 input_template):

        self.input_template = load_text(input_template)

    def construct_input(self, d_item, attack_method, defense_method, model=None, side="end"):
        d_item = attack_method(d_item, model=model, side=side)
        d_item = defense_method(d_item)
        user_input = self.input_template.format(instruction=d_item['instruction'], data=d_item['input'])
        return user_input

    def construct_multiturn_input(self, d_item, attack_method, defense_method, model=None):
        d_item = attack_method(d_item, model=model)
        d_item = defense_method(d_item)
        user_input = []
        for qa in d_item['additional_questions']:
            question = qa['question']
            answer = qa['answer']
            instruction = self.input_template.format(instruction=question, data=None)
            user_input += [{"role":"user", "content":instruction},
                           {"role":"assistant", "content":answer}]

        instruction = self.input_template.format(instruction=d_item['instruction'], data=d_item['input'])
        user_input += [{"role":"user", "content":instruction}]
        return user_input









if __name__ == '__main__':
    print(0)

