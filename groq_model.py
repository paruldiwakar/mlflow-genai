import mlflow.pyfunc
import pandas as pd
import mlflow
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains.llm import LLMChain
from mlflow.models import set_model
load_dotenv()

class GroqQAWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        system_prompt = "Answer the following question in two sentences"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{question}")
        ])

        llm = ChatGroq(
            model="moonshotai/kimi-k2-instruct-0905",
            temperature=0,
        )

        # LCEL chain is fine here
        self.chain = prompt | llm | StrOutputParser()

    def predict(self, context, model_input):
        import pandas as pd

        # Defensive: ensure chain exists (MLflow evaluate safety)
        if not hasattr(self, "chain"):
            self.load_context(context)

        # MLflow QA evaluators always pass "inputs"
        if "inputs" not in model_input.columns:
            raise ValueError(
                f"Expected column 'inputs'. Got {list(model_input.columns)}"
            )

        outputs = []
        for q in model_input["inputs"].tolist():
            outputs.append(
                self.chain.invoke({"question": q})
            )

        # MUST return list[str] or np.ndarray
        return outputs



# 👇 THIS IS THE KEY PART
set_model(GroqQAWrapper())