# corporate-mindfulness-mentor
## Iteration 1 Report – Goal Creation (User Story 1)

### Completed Stories:-
During this iteration, I worked on the **Goal Creation user story** for our *Corporate Mindfulness Mentor* project.  
The main objective was to allow users to enter a personal goal and duration (daily, weekly, or monthly) and receive a structured plan with mindfulness activities and an AI-generated summary.  

I implemented the backend logic using **LangGraph**, **LangChain / OpenAI API**, and connected it to the **Streamlit** interface.  
I also created and executed test scripts (`test_goal_creation.py`, `test_goal_creation_mock.py`) to validate functionality, mock API responses, and ensure all tests passed successfully.


### Deferred or Unfinished Stories:-
- The **Goal Decomposition** (sub-goal generation) feature was started but scheduled for completion in Iteration 2.  
- Some minor UI improvements, like dynamic activity refresh and better layout styling, were postponed for future iterations.


### Technical and Coordination Challenges:-
- Initially faced multiple **import path and PYTHONPATH issues** during pytest runs, which I resolved by restructuring the test imports and fixing relative paths.  
- Encountered **Streamlit interaction issues** — especially with button placement and non-clickable states — which were debugged and corrected.  
- Managed a few **dependency conflicts** within the virtual environment and worked closely with teammates to ensure smooth integration without merge conflicts.  
- Overall, focused on maintaining my part of the code independently while aligning with the team repository setup.


###  How the Prototype Works:-
1. The user enters a goal (e.g., *Reduce daily stress*), selects a duration, and optionally adds a short description.  
2. The system calls `run_goal_creation()` which triggers the **LangGraph workflow** and queries the **LLM** to generate activities and a summary in JSON format.  
3. The generated plan is displayed on the Streamlit interface, showing suggested activities and a short AI summary.  
4. Each plan can be saved locally in the `data/` folder and later extended using the **Decompose into Sub-goals** feature.


###  Reflection:-
This iteration helped me understand how to design and test an end-to-end AI workflow using LangGraph and Streamlit.  
I learned how to debug integration errors, manage structured responses from LLMs, and organize code for modular testing.  
The **Goal Creation feature** now runs smoothly and provides meaningful, structured output to users.  
Future work will focus on enhancing the goal decomposition logic and improving user experience.

