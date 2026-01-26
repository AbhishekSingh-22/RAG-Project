# Extraction Experiment Evaluation Report

## Summary of Averages (Scale 1-5)

| Strategy | Helpfulness | Specificity | Actionability |
|----------|-------------|-------------|---------------|
| visual_descriptive | 2.3 | 4.3 | 1.0 |
| functional_goal_oriented | 4.0 | 4.0 | 3.3 |
| structured_json | 1.3 | 1.7 | 1.0 |

## Detailed Results

### Image: HomeHawkApp_Users_Guide_CC1803YK9100_ENG_page_0001_img_001.png
**Query**: *How do I use the feature shown in HomeHawkApp Users Guide CC1803YK9100 ENG page 0001 img 001?*

- **visual_descriptive**: H=1 | S=4 | A=1
  - *Reasoning*: The description is highly specific to the image, accurately detailing text elements like 'Panasonic' and 'User's Guide' and layout features. However, the user is asking *how to use a feature* shown in the document, and the description only identifies the cover page/title page of the manual. It provides zero information about any actual feature, usage instructions, or content within the manual, making it completely unhelpful for answering the user's functional question.
- **functional_goal_oriented**: H=3 | S=3 | A=2
  - *Reasoning*: The user asked how to use a feature shown in the image, implying the image contains instructional content. The description correctly identifies the image as a cover page/title page (high specificity for what the image *is*), but this means the description *cannot* answer *how to use a feature*. The actionability is low because the only suggested actions are 'turn the page' or 'navigate to the Table of Contents,' which do not describe using a feature. The analysis is thorough for a title page, but ultimately unhelpful for the user's explicit question about a feature.
- **structured_json**: H=1 | S=2 | A=1
  - *Reasoning*: The retrieved description is extremely generic. It only identifies the document as the 'Panasonic® User's Guide' and provides a meaningless text entity ('en_us_201005'). It gives no information about the feature shown in the image, making it completely unhelpful and non-actionable for answering the user's question about how to use a specific feature on page 0001 of the guide.

---
### Image: HomeHawkApp_Users_Guide_CC1803YK9100_ENG_page_0006_img_001.png
**Query**: *How do I use the feature shown in HomeHawkApp Users Guide CC1803YK9100 ENG page 0006 img 001?*

- **visual_descriptive**: H=3 | S=5 | A=1
  - *Reasoning*: The description is highly specific and detailed, accurately mapping out the components and layout of the 'Overview' diagram (which is likely page 6 of the user guide). However, the user's question asks *how to use* a feature, and this description only provides an *overview* of the system components and connectivity ('What is the HomeHawk?'). It lists the devices but does not detail the operational steps required to 'use' any specific feature, making it low on actionability regarding the user's specific query.
- **functional_goal_oriented**: H=4 | S=4 | A=3
  - *Reasoning*: The description is very thorough and well-structured, effectively breaking down the likely purpose and function of the diagram based on the context implied by the HomeHawk system name. It directly addresses what the system is, how it functions (wireless network), and what the user should do next (informational absorption, then use the app). It scores high on helpfulness and specificity because it correctly identifies the diagram as a conceptual overview, although it cannot be perfectly specific without seeing the actual image. The actionability is moderate; while it suggests the implied next step (using the app), the image itself is an overview, not a direct instruction screen, which limits immediate, concrete action *from the image*. The analysis of the orange transmission lines is a good piece of inferred specificity.
- **structured_json**: H=1 | S=1 | A=1
  - *Reasoning*: The user is asking for instructions on how to use a specific feature on page 6, image 1 of a user guide. The retrieved description is a generic 'Overview' of 'What is the HomeHawk?' and provides high-level context about the camera system and its app. It contains no specific information about any feature or instruction relevant to page 0006, image 001, making it unhelpful and not actionable for the specific query.

---
### Image: HomeHawkApp_Users_Guide_CC1803YK9100_ENG_page_0007_img_001.png
**Query**: *How do I use the feature shown in HomeHawkApp Users Guide CC1803YK9100 ENG page 0007 img 001?*

- **visual_descriptive**: H=3 | S=4 | A=1
  - *Reasoning*: The description is highly specific to the content of page 7, detailing layout, note sections, and explicit text regarding device compatibility and registration limits, including visual descriptions of the hardware (Access point, Front door camera, Outdoor camera). However, the user's query asks *how to use* a feature. This page details *what* devices are compatible and *how many* can be registered, not the step-by-step *usage* instructions for a specific feature, making it only moderately helpful for a 'how-to' question. It is not actionable for 'how to use' beyond listing compatibility requirements.
- **functional_goal_oriented**: H=5 | S=5 | A=5
  - *Reasoning*: The description is highly helpful and actionable. It correctly interprets the image content (which must be a compatibility/setup diagram, given the analysis) as defining architecture, required components (Access Points), and system capacity limits (16 cameras/AP, 8 mobile devices). It clearly breaks down actions based on the user's goal (Setup, Adding Devices, Troubleshooting) and even preemptively addresses what limits will cause errors, effectively answering 'How do I use the feature?' by explaining the underlying system constraints and required setup.
- **structured_json**: H=2 | S=2 | A=1
  - *Reasoning*: The user is asking how to use a specific feature based on an image from page 7 of the manual. The retrieved description, however, details the *maximum number of devices* that can be registered (e.g., 16 front/outdoor cameras per access point, 8 mobile devices). This information is primarily about device capacity limits and does not explain *how to use* a feature, which is what the user asked for. Therefore, it is not very helpful for the stated goal, lacks specificity regarding a functional 'how-to' element, and offers no direct actionable steps for using a feature.

---
