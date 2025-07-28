## Trajectory Synthesis 

Below is the pseudocode for Explorer's web trajectory generation pipeline. The relevant prompts for all agents in the multi-agent pipeline are given in the paper Appendix.

Procedure Explorer(init_url):

  Initialize task_trajectory_data with:
    - actions ← ∅

  Set completed ← False

  Initialize:
    - task_refinement_history ← ∅
    - action_history ← ∅
    - step ← 0
    - execution_id ← 0

  While step < MAX_STEPS:
    action ← ∅

    If completed:
      Break

    If browser_env.page ≠ None:
      state ← GET_STATE()

      Save HTML and screenshots:
        - `html_step.html`
        - `screenshot_step.png`
        - `screenshot_som_step.png`

    Else:
      state ← None

    If step = 0:
      captcha_response ← CAPTCHA_AGENT.act(`screenshot_0.png`)
      If captcha_response = "yes":
        Return []

    If step = 0:
      task_proposal, action_nl, action_grounded, is_valid⟩ ← PROPOSAL_AGENT.act(state.a11y_tree, state.image_obs)
    Else:
      refined_goal, action_nl, action_grounded, , is_valid⟩ ← REFINER_AGENT.act(state.a11y_tree, state.image_obs, action_history, refined_goal)

    Update action:
      - step_action_nl ← action_nl
      - new_action_grounded ← action_grounded
      - step_refined_goal ← refined_goal OR task_proposal

    Append refined_goal to task_refinement_history
    Append action_nl to action_history

    If action_grounded = "stop":
      completed ← True
      Break

    If is_valid:
      action.URL_after ← browser_env.page.url
      Append action to task_trajectory_data.actions

    step ← step + 1

  screenshots ← ["screenshot_som_0.png", ..., "screenshot_som_{step}.png"]
  ⟨summ_response, user_intent⟩ ← SUMMARIZER.act(action_history, screenshots)

  history ← [a.step_action_nl ∀ a ∈ task_trajectory_data.actions]

  screenshots_all ← screenshots ∪ {final_screenshot}
  verifier_response ← VERIFIER.act(user_intent, history, screenshots_all, last_page_md)

  Update task_trajectory_data:
    - task_summary ← user_intent
    - verifier_agent_response ← verifier_response

  Return task_trajectory_data