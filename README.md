# AI Ethics Assignment - Group Project

## 📋 Project Overview

This repository contains our comprehensive analysis of AI ethics principles, case studies, and practical bias auditing. The assignment explores critical aspects of responsible AI development including algorithmic bias, transparency, fairness, and regulatory compliance.

## 👥 Contributors
| Name       | Role                | Contact Information | Responsibilities |
|------------|---------------------|---------------------|------------------|
| **Michael** | Software Developer | [@michael-wambua](slymike63@gmail.com) | Responsible for Part 3
| **Marion** | Software Developer | [@vtonbefore](beforevton@gmail.com) | Responsible for Part 2 and 4
| **Bati** | Software Developer | [@baatiroba2](bqunyo@gmail.com) | Responsible for Part 1


## 🎯 Assignment Structure

### Part 1: Theoretical Understanding (30%)
**Contributor: User 1**
- **Short Answer Questions**:
  - Q1: Define algorithmic bias with examples
  - Q2: Explain transparency vs explainability in AI
  - Q3: GDPR's impact on AI development in EU
- **Ethical Principles Matching**:
  - Justice, Non-maleficence, Autonomy, Sustainability

### Part 2: Case Study Analysis (40%)
**Contributor: User 1**
- **Case 1: Biased Hiring Tool** (Amazon AI recruiting)
  - Source of bias identification
  - Three fairness improvement proposals
  - Fairness evaluation metrics
- **Case 2: Facial Recognition in Policing**
  - Ethical risks analysis
  - Responsible deployment policies

### Part 3: Practical Audit (25%)
**Contributor: [Your Name]**
- **Dataset**: COMPAS Recidivism Dataset
- **Tools**: Python + AI Fairness 360 (IBM)
- **Deliverables**:
  - Bias analysis code
  - Visualizations (false positive rate disparities)
  - 300-word findings report
  - Remediation recommendations

### Part 4: Ethical Reflection (5%)
**Contributor: User 2**
- Personal project ethical analysis
- AI principles adherence strategies

### Bonus Task: Policy Proposal (Extra 10%)
**Contributor: User 2**
- **Topic**: Ethical AI use in healthcare
- **Requirements**:
  - Patient consent protocols
  - Bias mitigation strategies
  - Transparency requirements



## 🔧 Technical Requirements

### Part 3 - Bias Audit Setup
```bash
# Install required packages
pip install -r part3-bias-audit/requirements.txt

# Key dependencies:
# - aif360>=0.5.0
# - pandas>=1.3.0
# - numpy>=1.21.0
# - matplotlib>=3.5.0
# - seaborn>=0.11.0
# - scikit-learn>=1.0.0
# - tensorflow>=2.8.0
```

### Running the Bias Audit
```bash
cd part3-bias-audit/
python compas_bias_audit.py
```

## 📊 Key Deliverables

### Part 1 Deliverables (User 1)
- [x] Algorithmic bias definitions and examples
- [x] Transparency vs explainability analysis
- [x] GDPR impact assessment
- [x] Ethical principles matching exercise

### Part 2 Deliverables (User 1)
- [x] Amazon hiring bias analysis
- [x] Facial recognition ethics evaluation
- [x] Policy recommendations
- [x] Fairness metrics proposals

### Part 3 Deliverables ([Your Name])
- [x] **COMPAS bias audit code** - Complete Python analysis using AI Fairness 360
- [x] **Visualizations** - False positive rate disparities, confusion matrices, bias metrics
- [x] **300-word report** - Findings summary and remediation steps
- [x] **Bias mitigation** - Reweighing preprocessing implementation

### Part 4 Deliverables (User 2)
- [x] Personal project ethical reflection
- [x] AI principles adherence strategies

### Bonus Deliverables (User 2)
- [x] Healthcare AI ethics guidelines
- [x] Patient consent protocols
- [x] Bias mitigation strategies
- [x] Transparency requirements

## 📈 Part 3: Bias Audit Highlights

### Analysis Results
- **Dataset**: COMPAS Recidivism Dataset
- **Bias Type**: Racial bias in risk scoring
- **Key Findings**:
  - 23% higher false positive rates for African-American defendants
  - Disparate impact ratio: 0.647 (below 0.8 fairness threshold)
  - Significant statistical parity difference (>0.1)

### Visualizations Generated
1. **Recidivism Rate Comparison** - Actual vs predicted by race
2. **False Positive Rate Disparity** - Racial bias in error rates
3. **Confusion Matrices** - Performance breakdown by demographic
4. **Bias Metrics Dashboard** - Comprehensive fairness assessment

### Mitigation Strategies
- **Reweighing preprocessing** - Reduced disparate impact to 0.798
- **Adversarial debiasing** - In-processing fairness techniques
- **Post-processing calibration** - Error rate equalization

## 🎯 Assessment Criteria

| Component | Weight | Focus Areas |
|-----------|---------|-------------|
| Part 1 | 30% | Theoretical knowledge, concept clarity |
| Part 2 | 40% | Case analysis depth, solution quality |
| Part 3 | 25% | Technical implementation, findings quality |
| Part 4 | 5% | Reflection depth, practical application |
| Bonus | 10% | Policy comprehensiveness, innovation |

## 📋 Submission Checklist

- [x] All theoretical questions answered (Part 1)
- [x] Case studies analyzed with solutions (Part 2)
- [x] COMPAS bias audit completed (Part 3)
- [x] Code runs without errors
- [x] Visualizations generated successfully
- [x] 300-word report written
- [x] Ethical reflection completed (Part 4)
- [x] Healthcare policy proposal drafted (Bonus)
- [x] README documentation complete

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone [repository-url]
   cd ai-ethics-assignment
   ```

2. **Set up environment for Part 3**:
   ```bash
   cd part3-bias-audit
   pip install -r requirements.txt
   ```

3. **Run bias audit**:
   ```bash
   python compas_bias_audit.py
   ```

4. **Review outputs**:
   - Check console output for bias metrics
   - Review generated visualizations
   - Read bias audit report

## 📚 References

- **AI Fairness 360**: IBM's comprehensive toolkit for bias detection and mitigation
- **COMPAS Dataset**: Correctional Offender Management Profiling for Alternative Sanctions
- **GDPR**: General Data Protection Regulation documentation
- **Ethical AI Frameworks**: IEEE, ACM, and industry standards

## 🤝 Collaboration Guidelines

- **Communication**: Regular updates via [communication platform]
- **Code Review**: Cross-review technical implementations
- **Documentation**: Maintain clear, comprehensive documentation
- **Version Control**: Use meaningful commit messages and branch names
