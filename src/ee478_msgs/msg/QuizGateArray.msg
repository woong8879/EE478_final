std_msgs/Header header
# Quiz question text as read off the gate-pair scoreboard (e.g. "5+9=?").
string question
# Candidate gates. The drone must pass through whichever gate's label
# equals eval(question).
QuizGate[] gates
