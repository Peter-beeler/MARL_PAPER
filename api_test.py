import os
import re
from google import genai
from google.genai import types

api_key = os.getenv("VERTEX_AI_API_KEY")
if not api_key:
    print("Error: VERTEX_AI_API_KEY not found.")
    exit(1)

client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})

chat = client.chats.create(
    model='gemini-pro-latest',
    config=types.GenerateContentConfig(
        temperature=0.2,
    )
)

def generate_and_save_helpers(env_file_pth):
    # 1. Setup Client
    # 2. Read the environment file to give context to Gemini
    # We assume env_move.py is in the same directory
    try:
        with open(env_file_pth, "r") as f:
            env_content = f.read()
    except FileNotFoundError:
        print(f"Error: {env_file_pth} not found. Please make sure it is in the same directory.")
        return

    # 3. Construct the Prompt
    # We ask for specific XML tags <file name="..."> to make parsing easy
    prompt = f"""
    You are an expert Python developer.
    
    Here is the source code for a game environment file named `{env_file_pth}`:
    ```python
    {env_content}
    ```

    YOUR TASK:
    1. Analyze the environment code.
    2. Based on the format of agents' observations: it could be like grid symbols, or (x,y). Write a function in helpers.py that transforms these observations to language descriptions, inidicate the agent's global coordinates, and describes the objects and their coordinates in the observation. For example, "You are at (2,3). You see an apple at (2,4) and a dirt at (1,3) and agent 2 at (3,3)."
    3. Write a Python file named `helpers.py` containing high-level, callable functions to execute tasks (e.g., `move_to` to abstract the UP, DOWN, RIGHT, LEFT actions). You do not need to abstract every macro action, for example, fire you weapon could be a single function.
    Your function will be reused. Therefore, you should make it generic and reusable. Also, indicate in function return if this high-level task is done or not.
    This is an example function definition: def move_to(env, agent_id: int, coord_x, coord_y) -> bool. Agents could use json in responses to call these functions: For example: {{'action': 'move_to',  'agent_id': 1, 'args': {{'coord_x': 5, 'coord_y': 3}}}}, {{'action': 'eat',  'agent_id': 1}}.
    Each function should be as independent as possible, avoid calling another helper function inside a helper function and logic/movement overlapping. The purpose of these helpers is to be building blocks for the agent to call directly based on its observation and reasoning, not to call each other.
    4. Write a Markdown file named `helper_readme.md` explaining how to use these helpers.
    
    IMPORTANT OUTPUT FORMAT:
    You must wrap the content of each file in XML-style tags exactly like this:

    <file name="helpers.py">
    [Put the python code here]
    </file>

    <file name="helper_readme.md">
    [Put the markdown documentation here]
    </file>
    """

    print("Sending request to Gemini...")

    try:
        # 4. Generate Content via chat (history is managed automatically)
        response = chat.send_message(prompt)

        # 5. Parse the Response using Regex
        # Look for pattern: <file name="filename"> content </file>
        # flags=re.DOTALL allows the dot (.) to match newlines
        pattern = r'<file name="(.*?)">\s*(.*?)\s*</file>'
        matches = re.findall(pattern, response.text, flags=re.DOTALL)

        if not matches:
            print("Warning: Could not parse files from response. Raw output saved to 'gemini_raw_output.txt'")
            with open("gemini_raw_output.txt", "w") as f:
                f.write(response.text)
            return

        # 6. Save the files
        print(f"\nFound {len(matches)} files in response.")
        for filename, content in matches:
            # Remove any markdown code block fences if Gemini added them inside the tags
            content = content.replace("```python", "").replace("```markdown", "").replace("```", "").strip()
            
            with open(filename, "w") as f:
                f.write(content)
            print(f"Successfully saved: {filename}")

        print("\nDone! You can now import helpers in your code.")

    except Exception as e:
        print(f"\nGeneration Failed!")
        print(f"Error details: {e}")

def generate_prompt_for_agents():
    """Call Gemini to create a prompt template for high-level helper actions.

    Unlike grpo_text_action.py which uses low-level actions (up, down, left, right,
    clean, eat, stay), this generates a prompt template where the available actions
    are the high-level functions defined in helpers.py (e.g., smart_clean_step,
    smart_forage_step, move_to).
    """
    # 1. Setup Client

    # # 2. Read context files
    # try:
    #     with open("env_move.py", "r") as f:
    #         env_content = f.read()
    # except FileNotFoundError:
    #     print("Error: env_move.py not found.")
    #     return

    try:
        with open("helpers.py", "r") as f:
            helpers_content = f.read()
    except FileNotFoundError:
        print("Error: helpers.py not found.")
        return

    try:
        with open("helper_readme.md", "r") as f:
            helper_readme_content = f.read()
    except FileNotFoundError:
        print("Error: helper_readme.md not found.")
        return

    # 3. Construct the Prompt
    prompt = f"""
    You are an expert prompt engineer for LLM-based multi-agent reinforcement learning.

    You have seen the code of the game environment in env_move.py.
    ```

    HIGH-LEVEL HELPER FUNCTIONS:
    We have a `helpers.py` module with high-level action functions that abstract away
    the low-level movement logic. Here is the source code:
    ```python
    {helpers_content}
    ```

    And here is the documentation for these helpers:
    ```markdown
    {helper_readme_content}
    ```

    YOUR TASK:
    Suppose we already init an env class from env_move.py. You can use "env" as parameter if needed in function calls.
    Create a Python file named `prompt_template.py` that contains prompt-building
    functions for an LLM agent playing this game. The agent chooses from HIGH-LEVEL
    helper functions (not low-level actions like up/down/left/right).
    The prompt template must:
    1. Describe the game context derived from env_move.py: map layout, game rules,
       how apples spawn, how dirt works, the river/land zones, reward structure,
       and any other relevant mechanics you find in the environment code. For observation texts, describe the objects and their coordinates.
    2. List the available high-level actions with their signatures and usage
    3. Tell the agent to output ONLY ONE action function call name from high-level helpers (e.g., `smart_clean_step`, `smart_forage_step`, `move_to`) based on the current observation. Also define the response format so that I can easily parse it (e.g., just the function name as a string, or a JSON object with the function name and arguments).
    4. Provide two prompt-building functions:
       - `create_thinking_prompt(obs_text, agent_id)`: for two-stage generation,
         asks the agent to reason about the situation first.
       - `create_single_stage_prompt(obs_text, thinking_response, agent_id)`: for direct action output,
         asks the agent to output only one function call.
       Each function returns a list of message dicts with "role" and "content" keys
       (system/user roles) suitable for `tokenizer.apply_chat_template`.

    IMPORTANT OUTPUT FORMAT:
    Wrap the output in XML tags:

    <file name="prompt_template.py">
    [Put the python code here]
    </file>
    """

    print("Sending request to Gemini for prompt template generation...")

    try:
        # 4. Generate Content via chat (history is managed automatically)
        response = chat.send_message(prompt)

        # 5. Parse the Response
        pattern = r'<file name="(.*?)">\s*(.*?)\s*</file>'
        matches = re.findall(pattern, response.text, flags=re.DOTALL)

        if not matches:
            print("Warning: Could not parse files from response. Raw output saved to 'gemini_prompt_raw_output.txt'")
            with open("gemini_prompt_raw_output.txt", "w") as f:
                f.write(response.text)
            return

        # 6. Save the files
        print(f"\nFound {len(matches)} files in response.")
        for filename, content in matches:
            content = content.replace("```python", "").replace("```markdown", "").replace("```", "").strip()

            with open(filename, "w") as f:
                f.write(content)
            print(f"Successfully saved: {filename}")

        print("\nDone! Prompt template generated.")

    except Exception as e:
        print(f"\nGeneration Failed!")
        print(f"Error details: {e}")

def generate_agent_call_code(template_file_pth):
    """Generate code based on the template and high-level helper functions."""
    try:
        with open(template_file_pth, "r") as f:
            template_content = f.read()
    except FileNotFoundError:
        print(f"Error: {template_file_pth} not found.")
        return
    prompt = f"""
    You are an expert prompt engineer for LLM-based multi-agent reinforcement learning.

    You have seen the code of the game environment in env_move.py and created high-level helper functions in helpers.py and prompts for agents in prompt_template.py.

    YOUR TASK:
    Based on the template here:
    ```python
    {template_content}
    ```
    and helper functions and prompts, fill the template.

    IMPORTANT OUTPUT FORMAT:
    Wrap the output in XML tags:

    <file name="play.py">
    [Put the python code here]
    </file>
    """

    print("Sending request to Gemini for prompt template generation...")

    try:
        # 4. Generate Content via chat (history is managed automatically)
        response = chat.send_message(prompt)

        # 5. Parse the Response
        pattern = r'<file name="(.*?)">\s*(.*?)\s*</file>'
        matches = re.findall(pattern, response.text, flags=re.DOTALL)

        if not matches:
            print("Warning: Could not parse files from response. Raw output saved to 'gemini_prompt_raw_output.txt'")
            with open("gemini_prompt_raw_output.txt", "w") as f:
                f.write(response.text)
            return

        # 6. Save the files
        print(f"\nFound {len(matches)} files in response.")
        for filename, content in matches:
            content = content.replace("```python", "").replace("```markdown", "").replace("```", "").strip()

            with open(filename, "w") as f:
                f.write(content)
            print(f"Successfully saved: {filename}")

        print("\nDone! Prompt template generated.")

    except Exception as e:
        print(f"\nGeneration Failed!")
        print(f"Error details: {e}")
if __name__ == "__main__":
    generate_and_save_helpers("env_move.py")
    generate_prompt_for_agents()
    generate_agent_call_code("test_qwen4b_template.py")