from PyQt6.QtCore import QThread, pyqtSignal
import time
from rag_logic import rag_retriever # IMPORT RAG INSTANCE

# Import your actual RAG logic function
# from rag_logic import your_rag_function

class RAGWorker(QThread):
    """
    Threaded worker to handle RAG queries without freezing the UI.
    """
    signal_response_ready = pyqtSignal(str)

    def __init__(self, user_query, violations_list, rag_retriever):
        super().__init__()
        self.user_query = user_query
        self.violations_list = violations_list
        self.rag_logic_callback = rag_retriever

    def run(self):
        try:
            clean_violations = []

            for v in self.violations_list:
                if "FINAL SUMMARY" in v:
                    continue

                if "VIOLATION:" in v:
                    violation_name = v.split("VIOLATION:")[-1].strip()
                    clean_violations.append(violation_name)

            # Remove duplicates while preserving order
            unique_violations = list(dict.fromkeys(clean_violations))

            if not unique_violations:
                self.signal_response_ready.emit("No violations available for RAG.")
                return

            # 🔥 Send each violation separately
            for violation in unique_violations:

                print("Sending to RAG:", violation)
                violation = violation.replace("NO-", "").replace("-", " ").strip()

                # 🔥 Add context line
                violation = f"Worker not wearing {violation}. PPE safety guidelines."

                retrieved_docs = self.rag_logic_callback.retrieve(violation)

                if not retrieved_docs:
                    self.signal_response_ready.emit(
                        f"{violation} → No relevant safety documents found.\n"
                    )
                else:
                    # Print violation header once
                    self.signal_response_ready.emit(f"\nViolation: {violation}\n")

                    # Emit each document separately
                    for doc in retrieved_docs:
                        self.signal_response_ready.emit(f"- {doc['content']}\n")

                            # Emit each violation's response separately
                            #self.signal_response_ready.emit(response)

        except Exception as e:
            self.signal_response_ready.emit(f"RAG Error: {str(e)}")
