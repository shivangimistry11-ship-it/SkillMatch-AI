
from skill_extractor import extract_skills
from sklearn.metrics import precision_score, recall_score, f1_score

# Manually labeled test examples.
# The expected skills are the reference labels for each text.
test_cases = [
    {
        "text": "I develop applications using Python and Django.",
        "expected": {"Python", "Django"}
    },
    {
        "text": "Experienced in PHP, MySQL and RESTful APIs.",
        "expected": {"PHP", "MySQL", "REST API"}
    },
    {
        "text": "Machine learning with scikit-learn and pandas.",
        "expected": {"Machine Learning", "Scikit-learn", "Pandas"}
    },
    {
        "text": "CI/CD pipelines using Jenkins and Docker.",
        "expected": {"CI/CD", "Jenkins", "Docker"}
    },
    {
        "text": "Developing apps in Android Studio using Java.",
        "expected": {"Android Studio", "Java"}
    }
]

# Build a vocabulary from the expected labels.
all_skills = sorted({
    skill
    for case in test_cases
    for skill in case["expected"]
})

y_true = []
y_pred = []

print("\nSKILLMATCH AI - EVALUATION RESULTS\n")

for index, case in enumerate(test_cases, start=1):
    detected = set(extract_skills(case["text"]))

    print(f"Test {index}")
    print("Expected:", sorted(case["expected"]))
    print("Detected:", sorted(detected))
    print("Correct:", sorted(case["expected"] & detected))
    print()

    for skill in all_skills:
        y_true.append(int(skill in case["expected"]))
        y_pred.append(int(skill in detected))

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("Overall Evaluation")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1-score:  {f1:.2%}")
