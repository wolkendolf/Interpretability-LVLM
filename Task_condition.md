# Multimodal LLMs - test task
Analyze the performance of multimodal models using ideas from the TransformerLens library. To complete the task, you can use Google Colab or Kaggle Notebooks computing resources. Prepare code and a report with a detailed description of the experiments and results. You have **7 days** from the date of receipt to complete the task.
Steps for completing the task:
1. Study the Logit Lens approach
  - Understand how the Logit Lens method works.
  - Briefly describe what Logit Lens is and how this approach helps analyze models.
4. Apply the framework to multimodal models
  - Analyze selected models using the logit lens approach. You can add to the transformerslens framework or implement the approach yourself. You can choose the model type and size yourself (for example, take llava-onevision at 1.5b or qwen2-vl at 2b).
  - Select several datasets for testing (for example, images with different objects, colors, shapes, etc.). These can be benchmarks or any other datasets.
- Explore the dynamics of logits at different stages of the model's operation.
  - Try to identify interesting patterns, such as the appearance or disappearance of features (e.g., colors, shapes, textures).
  - Describe what you found: what changes occur in the logs? In which model blocks do they occur?
10. Analysis of problems and proposed solutions

Analyze the problems or behavioral characteristics of the models that you have identified:
- Suggest possible ways to solve the problems you have identified.
- Describe the strengths and weaknesses of the proposed approaches.

Next, you can conduct research on existing scientific works that describe similar problems in multimodal models:
- Compare the approaches from the works with your own.
- Describe the differences in approaches and ways of solving the problem.
- Think about how you can combine your approach with the solutions you have found to improve the result.

Tips for completing the task:
- Structure your code and conclusions so that they are easy for the reviewer to understand.
- Use visualizations to clearly present your results.
  
If you encounter difficulties, you can ask ChatGPT, but the main thing is to understand and be able to explain what was done.
Rules:
- Analyze the results. This is the most important point because we want to see more than just numbers with metrics. How do you explain the behavior you observed? Write a report on the experiments you conducted. What did you find? What didn't work?
- There is no right way to solve the problem. Don't worry that you are doing something wrong. We want to see your research skills, not a specific solution to the problem. Perhaps you will come up with something we didn't even think of initially — that would be awesome.
- Make sure the results are reliable. Rule out the possibility of chance, etc.
- Send your solution as a repository on GitHub with a report on the solution and clear instructions on how to run your code. Make sure we can run your solution using these instructions.
- You can use any libraries and frameworks you may need.
- Focus on keeping the code clean and understandable. If you think any part of it may be unclear, add comments. We greatly appreciate well-written code, so if the solution to the problem is messy, we may reject the application.
