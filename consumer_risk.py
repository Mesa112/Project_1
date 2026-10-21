from pathlib import Path
from instruct_models import Instruct_Model

class Consumer_Risk:
    """Progam that takes a input text and returns if there might be a risk or not with given text"""
    def __init__(self):
        self.text = None

        self.instruct_model = Instruct_Model()

    def run_program(self):
        curr_contract_path = Path(__file__).resolve().parent / "contracts" /  "contract_1.txt"
        content = self.input_text(curr_contract_path)
        self.instruct_model.load_instruct_model()
        results = self.Analyze_risk(content)
        print(results)

    def input_text(self, addr):
        self.text = Path(addr)
        with self.text.open('r') as file:
            content = file.read()
            content = content.strip()
            content = content.lower()

        return content

    def Analyze_risk(self, content):
        risk_found = self.instruct_model.instruct_model_param(content)
        response = self.instruct_model.instruct_model_output(risk_found)

        return response

if __name__ == "__main__":
    program = Consumer_Risk()
    program.run_program()