from transformers import pipeline
import torch


class Instruct_Model:
    def __init__(self):
        self.pipeline = None

    def load_instruct_model(self):
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        self.pipeline = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype" : torch.bfloat16},
            device_map="auto",
        )
    def instruct_model_param(self, contract_txt):
        messages = [
        {
        "role": "system",
        "content": """
        You are a consumer contract analysis assistant.

        Your task is to analyze contracts from the consumer's
        perspective and identify terms that may affect the
        consumer's finances, rights, obligations, or flexibility.

        Do not produce only a general summary. Instead, provide
        a detailed analysis organized into three categories:

        1. Risks:
           Identify potentially unfavorable terms. Quote the
           relevant contract text, explain it in plain English,
           and describe its potential consequences.

        2. Unanswered questions:
           Identify ambiguous terms, missing information, or
           provisions whose consequences cannot be determined
           from the provided text. Explain what needs clarification
           and why it matters.

        3. Benefits:
           Identify protections, guarantees, rights, or other
           advantages explicitly supported by the contract.

        Rules:
        - Base every finding on the provided contract.
        - Include the exact supporting text as evidence.
        - Preserve relevant conditions, exceptions, and limitations.
        - Do not invent terms, facts, benefits, or consequences.
        - Distinguish explicit contract terms from potential
          implications or uncertainties.
        - Do not assume an ambiguous term is automatically
          unenforceable.
        - Avoid repeating the same finding unnecessarily.
        - If a category has no findings, return an empty list.

        Return the analysis as valid JSON with three top-level keys:
        "risks", "unanswered_questions", and "benefits".
        Do not replace the detailed analysis with a general summary.
        CLASSIFICATION RULES:

        The categories are NOT mutually exclusive.

        For every clause, independently evaluate:

        1. RISK:
        Does this clause create a potentially unfavorable
        financial obligation, restriction, or consequence?

        2. UNANSWERED QUESTION:
        Does this clause omit material information needed
        to understand its costs, conditions, or consequences?

        3. BENEFIT:
        Does this clause explicitly provide a consumer
        protection, right, or advantage?

        A clause may produce findings in multiple categories.

        Do not treat identifying a risk as sufficient
        if important information about that risk is missing.

        Do not create unanswered questions about information
        already explicitly stated in the contract.

        Check the full supplied contract for answers
        before reporting missing information.
        
        """,
    },
    {
        "role" : "user",
        "content" : f"Analyze this contract:\n\n{contract_txt}",
            },
    ]
        return messages
    def instruct_model_output(self, messages):
        if self.pipeline is None:
            raise RuntimeError("No model loaded for output")
        output = self.pipeline(
            messages,
            max_new_tokens=8000 ,
            do_sample=False
        )

        response = output[0]["generated_text"]

        return response[-1]["content"]



