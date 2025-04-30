import pandas as pd
import dspy
import os
from dspy.evaluate import Evaluate
from dspy.evaluate.metrics import answer_exact_match

# Load the CSV dataset
df = pd.read_csv("shell_commands_dataset.csv")

# Shuffle the dataset
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Convert to list of dictionaries
trainset_dict = df_shuffled.to_dict(orient='records')

# Split dataset (80% train, 20% test)
split_index = int(len(trainset_dict) * 0.8)
trainset = trainset_dict[:split_index]
testset = trainset_dict[split_index:]

# Create examples with 'answer' instead of 'command'
train_examples = [
    dspy.Example(question=x["prompt"], answer=x["command"]).with_inputs("question")
    for x in trainset
]
test_examples = [
    dspy.Example(question=x["prompt"], answer=x["command"]).with_inputs("question")
    for x in testset
]

# Updated Signature using 'answer'
class resultantcommand(dspy.Signature):
    """Generate shell commands from natural language prompts."""
    question = dspy.InputField(desc="User's natural language prompt")
    answer = dspy.OutputField(desc="Correct shell command sequence")

class CommandGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(resultantcommand)
    
    def forward(self, question):
        return self.generate(question=question)

# Configuration (use environment variable for API key)
api_key = os.getenv('OPENAI_API_KEY') or 'your-api-key-here'
lm = dspy.OpenAI(model='gpt-3.5-turbo', api_key=api_key)
dspy.configure(lm=lm)

# Initialize and evaluate
generator = CommandGenerator()

evaluator = Evaluate(
    devset=test_examples,
    metric=answer_exact_match,  # Now works with default 'answer' field
    display_progress=True,
    display_table=5,
    num_threads=4
)

results = evaluator(generator)
print(f"\nEvaluation Results - Exact Match Accuracy: {results * 100:.2f}%")