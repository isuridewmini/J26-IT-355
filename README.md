# AI-Powered Adaptive Learning System for Children with Intellectual Disabilities

## About the Project

The **AI-Powered Adaptive Learning System for Children with Intellectual Disabilities** is an intelligent learning application designed to support personalised learning experiences for children with different learning needs.

The system combines Artificial Intelligence (AI) and Machine Learning (ML) to screen for potential learning difficulties, identify learning preferences, generate suitable learning activities, and monitor children's engagement during learning sessions.

The system consists of four interconnected components. Each component performs a specific task and shares relevant information with the other components to support adaptive learning.

## System Components

### Component 1: AI-Based Learning Difficulty Screening

**Purpose:** To screen for potential autism-related or dyslexia-related learning difficulties using information provided by a parent, teacher, or caregiver.

Key functions:
- Collects relevant child information and behavioural or learning observations.
- Uses trained ML models to estimate risk for the selected screening condition.
- Uses a 10-question follow-up questionnaire to classify the level of difficulty as Mild, Moderate, or Severe when the child is flagged as at risk.
- Provides screening results and guidance.
- Sends the difficulty profile and severity level to Component 3.

**Technologies:** Flutter, Dart, Gradient Boosting, Logistic Regression.

**Important:** The component provides screening support, not a medical diagnosis.

### Component 2: AI-Based Visual and Auditory Learning Preference Identification

**Purpose:** To evaluate a child's performance in visual and auditory learning activities and identify the learning presentation format in which the child performs better.

Key functions:
- Receives the learning difficulty information from Component 1.
- Presents visual and auditory learning activities.
- Evaluates responses using XGBoost and Gemini AI.
- Generates separate performance scores for the two learning formats.
- Sends the predicted learning preference to Component 3.

**Technologies:** Machine Learning, XGBoost, Gemini AI.

**Output:** A predicted learning preference, such as Visual or Auditory, based on the assessment results.

### Component 3: AI-Powered Adaptive Learning Activity Generation

**Purpose:** To generate and deliver personalised Mathematics and English learning activities based on the child's learning needs, difficulty profile, severity level, and preferred learning presentation format.

Key functions:
- Receives the difficulty profile from Component 1 and learning preference from Component 2.
- Determines a target reading level for learning questions.
- Uses AI to generate age- and difficulty-appropriate questions.
- Checks the readability of generated questions using an on-device model.
- Revises questions that do not meet the target reading level and flags unsuitable questions for teacher review.
- Delivers learning activities through visual or audio formats.
- Evaluates the child's answers and provides feedback.
- Sends the child ID, session ID, answers, and assessment verdicts to Component 4.

**Key principle:** Generated questions are checked for readability before being presented to the child.

### Component 4: AI-Based Real-Time Engagement and Emotional State Detection for Adaptive Feedback

**Purpose:** To estimate a child's engagement level during learning activities using camera-based observations and provide an engagement summary to support personalised learning and teacher review.

Key functions:
- Receives the child ID and session ID associated with the learning activity.
- Captures consent-authorised camera observations during a learning session.
- Processes observations on the device to estimate engagement probabilities.
- Classifies estimated engagement as Low, Moderate, or High.
- Identifies unavailable or insufficient evidence instead of guessing when observations are missing or unreliable.
- Generates a session summary containing average engagement, observation coverage, and uncertainty information.
- Sends the engagement summary to Component 3 to support adaptive feedback and teacher review.

**Important:** Engagement estimates are indicators, not definitive measurements of a child's internal emotional state. The component supports teachers and does not replace their judgement.

## How the Components Work Together

1. **Component 1 — Screening:** Estimates potential learning difficulties and provides a difficulty profile.
2. **Component 2 — Learning preference:** Evaluates visual and auditory performance to identify the child's likely preferred learning format.
3. **Component 3 — Adaptive learning:** Generates, checks, and delivers personalised learning activities and evaluates answers.
4. **Component 4 — Engagement monitoring:** Estimates engagement during learning sessions and returns a summary to support adaptive feedback.

The components exchange relevant information to help personalise learning activities while keeping teachers involved in reviewing results and supporting each child.

## Example Learning Workflow

1. A parent or caregiver completes the initial screening process.
2. The system provides a screening result and, when applicable, a follow-up severity classification.
3. The child completes visual and auditory assessment activities.
4. The system estimates the child's learning presentation preference.
5. Component 3 generates an appropriate Mathematics or English question and checks its reading level.
6. The child completes the learning activity while Component 4 monitors engagement when camera observation is authorised.
7. Component 3 receives the engagement summary and uses the available evidence to support adaptive feedback.
8. The teacher can review uncertain results and provide additional support where necessary.

## Privacy, Safety, and Limitations

- Obtain appropriate parental or guardian consent before collecting children's information or using camera observations.
- Treat screening results as estimates rather than medical diagnoses.
- Process camera observations on the device and delete temporary images after processing, according to the implemented data-handling protocol.
- Do not interpret a missing face or poor-quality observation as evidence of disengagement.
- Provide an alternative learning experience when camera access is unavailable.
- Use an insufficient-evidence status when there is not enough reliable information to produce an engagement summary.
- Protect children's information and restrict access to authorised users.
- Treat AI-generated content, screening estimates, learning preferences, and engagement predictions as decision-support information requiring appropriate oversight.

## Project Goals

- Support personalised learning for children with diverse learning needs.
- Adapt Mathematics and English activities to appropriate reading levels.
- Explore AI-based identification of visual and auditory learning preferences.
- Monitor engagement during learning sessions where consent and reliable observations are available.
- Provide useful information to teachers for reviewing learning progress and adapting support.

## Repository Structure

The repository will contain the implementation and documentation for the four system components.

As development progresses, component-specific source code, model files, setup instructions, and documentation will be organised into appropriate directories.

## Disclaimer

This project is an educational support system under development. Its outputs are intended to assist learning and teacher decision-making, not to replace professional assessment, medical diagnosis, or teacher judgement.
