def test_goal_creation_with_valid_inputs():
       result = run_goal_creation("reduce stress", "daily", "office work")
       assert result.goal == "reduce stress"
       assert len(result.suggested_activities) >= 3