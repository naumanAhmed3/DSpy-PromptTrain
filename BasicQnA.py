import os
import dspy
import dspy.predict
lm = dspy.LM('openai/gpt-3.5-turbo', api_key=os.environ["OPENAI_API_KEY"])
dspy.configure(lm=lm)


#
# class QA(dspy.Signature):
#     input = dspy.InputField(desc = "Any Question")
#     jawab = dspy.OutputField(desc="Often between 1 and 5 words")

# predict = dspy.ChainOfThought(QA)

# print(predict(input = "Mumtaz Qadri's father's profession").jawab)




class dooublechainofthought(dspy.Module):
    def __init__(self):
        self.firstChainofThought = dspy.ChainOfThought("question -> detailed_thought")
        self.secondChainofThought = dspy.ChainOfThought("question, detailed_thought -> one_word_answer")

    def forward(self,question):
        thought = self.firstChainofThought(question=question).detailed_thought
        answer = self.secondChainofThought(question=question , detailed_thought=thought).one_word_answer
        return dspy.Prediction(thought=thought , answer=answer)
    



double_cot = dooublechainofthought()
print(double_cot(question = "What was Job of father of man who killed salman tasir"))











