import os
import dspy
import dspy.predict
from pydantic import BaseModel ,  Field 

lm = dspy.LM('openai/gpt-3.5-turbo', api_key=os.environ["OPENAI_API_KEY"])
dspy.configure(lm=lm)

class answer_form(BaseModel):
    answer: str=Field("Answer between 1 to 20 words")
    confidence: float=Field("Your Connfidence about the answer between 0 and")



class answer_with_confidence(dspy.Signature):
    question: str = dspy.InputField(desc="Any random question")
    thought: str = dspy.InputField(desc="The step by step thought produced")
    final_answer: answer_form=dspy.OutputField() 


class doublechainofthought(dspy.Module):
    def __init__(self):
        self.cot1 = dspy.ChainOfThought("question -> detailed_thought")
        self.cot2 = dspy.ChainOfThought(answer_with_confidence)
    
    def forward(self , question):
        print("forward method is invoked")
        thought = self.cot1(question=question).detailed_thought
        final_asnwer = self.cot2(question=question,thought=thought)
        return final_asnwer


cot = doublechainofthought()
print(cot.forward("What are the names of people who were standing with salman tasir while his murder and also name the job of father of person who killed salman tasir"))




