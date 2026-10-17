"""route user requests through three document tools"""
import json
from ollama import Message, chat
from source.semantic import load_documents
from source.rag import MODEL_NAME, answer_question

SYSTEM_PROMPT = """ You are the Fetchdex AI Assistant.
Use grounded_answer for every question about document contents.
Use list_documents when the user asks which documents exist.
Use read_document when the user requests a complete file by exact filename.
Never invent document contents or answer document questions from memory.
After receiving tool results, answer concisely and preserve source citations.
You may answer directly only when explaining how to use Fetchdex.
"""

# -----------------------------------------------------------------------------------------------------------------------------
# TOOL FUNCTIONS
# these are the Python functions the language model is allowed to choose

# TOOL: LLM chooses RAG flow
def grounded_answer(question: str, k: int = 3) -> str:
    """answer a question using semantic search, document evidence and citations"""
    # run the complete RAG flow and return only its final answer text
    return answer_question(question, k)["answer"]

# TOOL: LLM chooses to list documents
def list_documents() -> str:
    """return the names of all available text documents"""
    # load every document and keep only its filename
    names = [document["source"] for document in load_documents()]
    # convert the Python list into readable JSON text for the model
    return json.dumps(names, indent=2)

# TOOL: LLM chooses to read documents
def read_document(name: str) -> str:
    """return one complete document matched by its exact filename"""
    # reject a missing or whitespace-only filename
    if not name.strip(): return "Document name cannot be empty."
    # check each loaded document
    for document in load_documents():
        # return the complete text when the filename matches
        if document["source"] == name: return document["text"]
    # tell the model when no document matched
    return f"Document '{name}' was not found."

# expose the three Python functions to ollama as available tools
TOOLS = [grounded_answer, list_documents, read_document]
# connect each tool name returned by the model to the actual Python function
TOOL_MAP = {tool.__name__: tool for tool in TOOLS}

# -----------------------------------------------------------------------------------------------------------------------------

# (MAIN) assistant function: model chooses which tool is best
def assistant_search(user_message: str, max_steps: int = 5) -> str:
    """let the model select and run tools until it can return a final answer
       max_steps prevents the assistant from calling tools forever"""
    # 1. reject an empty or whitespace-only user message
    if not user_message.strip(): raise ValueError("Message cannot be empty.")

    # 2. create and begin the conversation with LLM
    messages: list[Message] = [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=user_message),
    ]

    for _ in range(max_steps):
        try:
            # send the conversation and available tools to LLM
            response = chat(model=MODEL_NAME, messages=messages, tools=TOOLS)

        # explain model/server failures as one clear application error
        except Exception as error: raise RuntimeError(f"Local LLM unavailable: {error}") from error

        # 4. preserve the model's response in the conversation history
        messages.append(response.message)

        # 5. return the response when the model does not request a tool
        if not response.message.tool_calls:return response.message.content or "The assistant returned an empty response."

        # 6. execute every tool requested by the model
        for call in response.message.tool_calls:
            # read the requested function name
            name = call.function.name
            # find the matching Python function
            function = TOOL_MAP.get(name)

            # handle a tool name that Fetchdex does not provide
            if function is None: result = f"Unknown tool: {name}"
            else:
                # unpack the model's arguments and call the selected function
                try: result = function(**call.function.arguments)
                # return the tool error to the model so it can respond safely
                except Exception as error: result = f"Tool '{name}' failed: {error}"

            # RAG already created the final cited answer, so return it directly
            if name == "grounded_answer" and function is not None: return str(result)

            # add other tool results to the conversation for the model to explain
            messages.append(Message(role="tool", tool_name=name, content=str(result)))

    # 7. stop safely if the model used every allowed tool step
    return "The assistant reached its tool-step limit before producing an answer."

# -----------------------------------------------------------------------------------------------------------------------------
# COMMAND-LINE ENTRY POINT

# def main() -> None:
#     """
#     run one assistant question from the command line
#     """
#     # import command-line parsing only when this entry point is used
#     import argparse

#     # create the command-line parser
#     parser = argparse.ArgumentParser(description="Ask the Fetchdex AI Assistant.")
#     # require one question or document request
#     parser.add_argument("question", help="Question or document request")
#     # read the user's command-line input
#     args = parser.parse_args()
#     # run the main assistant flow and print its answer
#     print(assistant_search(args.question))


# run main() only when this file is executed as a module
# if __name__ == "__main__":
#     main()
