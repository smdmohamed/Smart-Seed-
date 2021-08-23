# Smart-Seed-
Smart Seed was a project proposed in first edition of the Algeria AI challenge. 

This work was done as part of the participation in the first edition of the Algeria AI challenge, This work was done by an Algerian student team from the Ecole Nationale Polytechnique:

Abdel-Malek Atchi: Electrical Engineering Dpt.

Mohamed Sammad: Mechanical Engineering Dpt.

Mosaab Derba: Control Engineering Dpt.

Sara Messara: Control Engineering Dpt.

Under the supervision of:

Sofian Chetoui: Electrical and Computer Engineering group, Brown University

Description:


The proposal file is the proposed project of the first round of the competition, it proposed the implementation of an edge-based AI application for Smart management of Irrigation and weed detection. The project meets positive feedback from the committee board and reaches the final stage of the competition.
unfortunately, due to the disruptive events caused by the COVID-19 outbreak, and after the revise of the board committee to reduce the scope of the project, the project was changed to a plan classification AI App based on leaf scanning.

Dataset:

We have used Flavia data set, more information can be found on: http://flavia.sourceforge.net/

Code:

Our code was written in Python using TensorFlow as a framework.

data_labeling.py: produce the pickle format of the training and testing data set

main.py: contains the core of the classification algorithm with a simple architecture.

VGG16.py: is the implementation of VGG16 architecture on our dataset.

Restnet50.py: is the implementation of Restnet50 architecture on our dataset.

