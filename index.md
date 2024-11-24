---
layout: project_page
permalink: /

title: 
    基于大语言模型的机械臂操作
authors:
    王东
affiliations:
    大连理工大学 控制科学与工程学院
# paper: https://www.cs.virginia.edu/~robins/Turing_Paper_1936.pdf
# video: https://www.youtube.com/watch?v=arhAzLl3jJc
# code: https://github.com/topics/turing-machines
---


<div class="content has-text-justified">
	为深入贯彻国家教育数字化战略行动的实施，加速推进人工智能在研究生教育领域的创新应用，积极探索并推广适应新时代需求的研究生教育教学新形态、新模式，我们致力于推动人工智能技术与学术实践的深度融合。

我们的目标是培养既具备人工智能理论和实践能力，又精通行业应用场景的复合型创新人才，为社会和产业注入新动能。基于此，我们特别聚焦于当前利用大语言模型（LLM）实现机械臂操作的前沿方法，进行了系统性调研，并努力搜集和整理相关的开源程序资源。这一举措旨在为学生提供亲身体验人工智能技术带来的前所未有的变革机会，让他们在实际操作中更好地理解人工智能的潜力和价值。

未来，我们将继续推动相关领域的深度研究，打造更加开放、前沿、创新的学习环境，帮助研究生在人工智能驱动的新时代中实现全面发展，为国家科技创新和经济高质量发展贡献力量。
        </div>

## 文献一
<!-- Using HTML to center the abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>    </h2>
        <h2>VLMPC: Vision-Language Model Predictive Control for Robotic Manipulation</h2>
        <h2 style="font-size: 16px;">作者：Wentao Zhao*, Jiaming Chen*, Ziyu Meng, Donghui Mao, Ran Song†, Wei Zhang</h2>
        <h2 style="font-size: 16px;">单位：山东大学</h2>
        <h2 style="font-size: 16px;">Robotics: Science and Systems (RSS), 2024.</h2>
        <div class="content has-text-justified">
摘要: 尽管模型预测控制（MPC）可以有效地预测系统的未来状态，因此被广泛应用于机器人操纵任务中，但它不具有环境感知能力，导致在一些复杂的场景中失败。为了解决这个问题，我们引入了视觉语言模型预测控制（VLMPC），这是一种机器人操纵框架，利用视觉语言模型（VLM）的强大感知能力，并将其与MPC集成。具体来说，我们提出了一种条件动作采样模块，该模块将目标图像或语言指令作为输入，并利用VLM对一组候选动作序列进行采样。然后，设计了一个轻量级的动作条件视频预测模型，以生成一组基于候选动作序列的未来帧。VLMPC通过分层成本函数在VLM的帮助下产生最优动作序列，该函数规定了当前观测值和目标图像之间的像素级和知识级一致性。我们证明VLMPC在公共基准测试中优于最先进的方法。更重要的是，我们的方法在机器人操纵的各种现实任务中表现出色。代码可在以下网址获得https://github.com/PPjmchen/VLMPC.
        </div>
    </div>
</div>




---
![Turing Machine](/static/image/1vlm.jpeg)

*图 1: VLMPC接收目标图像或语言指令作为输入。它首先提示VLM生成条件采样分布，从中推导出动作序列。然后，这些动作序列被馈送到轻量级的动作条件视频预测模型中，以预测一组未来的帧。VLMPC的评估是通过由两个子成本组成的分层成本函数进行的：像素距离成本和基于未来帧进行视频评估的VLM辅助成本。VLMPC最终选择最佳动作序列，其中机器人选择要执行的第一个动作，随后的动作被输入动作采样模块，以进一步辅助条件动作采样。[文献链接](https://arxiv.org/abs/2407.09829)&[项目链接](https://github.com/PPjmchen/VLMPC)*





## 文献二
<!-- Using HTML to center the abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>    </h2>
        <h2>Do As I Can, Not As I Say: Grounding Language in Robotic Affordances</h2>
        <h2 style="font-size: 16px;">作者：1 Michael Ahn∗
, Anthony Brohan∗
, Noah Brown∗
, Yevgen Chebotar∗
, Omar Cortes∗
, Byron David∗
,
Chelsea Finn∗
, Chuyuan Fu†
, Keerthana Gopalakrishnan∗
, Karol Hausman∗
, Alex Herzog†
,
Daniel Ho†
, Jasmine Hsu∗
, Julian Ibarz∗
, Brian Ichter∗
, Alex Irpan∗
, Eric Jang∗
,
Rosario Jauregui Ruano∗
, Kyle Jeffrey∗
, Sally Jesmonth∗
, Nikhil J Joshi∗
, Ryan Julian∗
,
Dmitry Kalashnikov∗
, Yuheng Kuang∗
, Kuang-Huei Lee∗
, Sergey Levine∗
, Yao Lu∗
, Linda Luu∗
,
Carolina Parada∗
, Peter Pastor†
, Jornell Quiambao∗
, Kanishka Rao∗
, Jarek Rettinghouse∗
,
Diego Reyes∗
, Pierre Sermanet∗
, Nicolas Sievers∗
, Clayton Tan∗
, Alexander Toshev∗
,
Vincent Vanhoucke∗
, Fei Xia∗
, Ted Xiao∗
, Peng Xu∗
, Sichun Xu∗
, Mengyuan Yan†
, Andy Zeng∗
</h2>
        <h2 style="font-size: 16px;">单位：∗Robotics at Google, †Everyday Robots</h2>
        <h2 style="font-size: 16px;">6th Conference on Robot Learning (CoRL 2022)</h2>
        <div class="content has-text-justified">
摘要：大型语言模型可以编码关于世界的丰富语义知识。这些知识对于旨在根据自然语言表达的高级、时间扩展的指令行事的机器人来说可能非常有用。然而，语言模型的一个显著弱点是它们缺乏现实世界的经验，这使得在给定的实施例中很难利用它们进行决策。例如，要求语言模型描述如何清理泄漏可能会导致合理的叙述，但它可能不适用于需要在特定环境中执行此任务的特定代理，如机器人。我们建议通过预训练的技能提供现实世界的基础，这些技能用于约束模型提出可行且符合上下文的自然语言动作。机器人可以充当语言模型的“手和眼睛”，而语言模型提供有关任务的高级语义知识。我们展示了低级技能如何与大型语言模型相结合，以便语言模型提供有关执行复杂和时间扩展指令的过程的高级知识，而与这些技能相关的值函数则提供了将这些知识与特定物理环境联系起来所需的基础。我们在许多现实世界的机器人任务上评估了我们的方法，在这些任务中，我们展示了对现实世界接地的需求，并且这种方法能够在移动操纵器上完成长时间、抽象、自然语言的指令。该项目的网站、视频和桌面域中的开源代码可以在https://say-can.github.io/上找到。
        </div>
    </div>
</div>

---
![Turing Machine](/static/image/2doAsICan.png)

*图 2: 我们可视化了SayCan的决策过程。蓝色条表示（标准化的）LLM概率，红色条表示（标准化的）选定技能成功执行的概率。组合得分以绿色条显示，算法选择得分最高的技能。这种可视化强调了SayCan的可解释性。对于任务“我打翻了可乐，你能给我带点东西来清理吗？”，SayCan成功规划并执行了以下步骤：1. 找到一个海绵 2. 拿起海绵 3. 把它带给你 4. 完成。[文献链接](https://arxiv.org/abs/2204.01691)&[项目链接](https://say-can.github.io/)*




## 文献三
<!-- Using HTML to center the abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>    </h2>
        <h2>VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models</h2>
        <h2 style="font-size: 16px;">作者：Wenlong Huang1
, Chen Wang1
, Ruohan Zhang1
, Yunzhu Li1,2
, Jiajun Wu1
, Li Fei-Fei1
</h2>
        <h2 style="font-size: 16px;">单位：Stanford University，University of Illinois Urbana-Champaign</h2>
        <h2 style="font-size: 16px;">7th Conference on Robot Learning (CoRL 2023)</h2>
        <div class="content has-text-justified">
摘要：大型语言模型（LLMs）被证明拥有丰富的可操作知识，可以以推理和规划的形式提取用于机器人操纵。尽管取得了进展，但大多数仍然依赖于预定义的运动图元来与环境进行物理交互，这仍然是一个主要的瓶颈。在这项工作中，我们的目标是在给定一组开放指令和一组开放对象的情况下，为各种操纵任务合成机器人轨迹，即一系列密集的6-DoF末端执行器航路点。我们通过首先观察LLM在自由形式语言教学中擅长推断启示和约束来实现这一点。更重要的是，通过利用他们的代码编写能力，他们可以与视觉语言模型（VLM）交互，组成3D价值图，将知识融入代理的观察空间。然后，在基于模型的规划框架中使用合成的值映射来合成具有对动态扰动的鲁棒性的零样本闭环机器人轨迹。我们进一步展示了所提出的框架如何通过有效地学习涉及联系人丰富交互的场景的动态模型，从在线体验中受益。我们在模拟和真实机器人环境中对所提出的方法进行了大规模研究，展示了以自由形式的自然语言执行各种日常操作任务的能力。视频和代码请访问https://voxposer.github.io/。
        </div>
    </div>
</div>

---
![Turing Machine](/static/image/3Voxposer.jpeg)

*图 3: 根据环境的RGB-D观察和语言指令，LLM生成代码，与VLM互动，产生一系列3D可赋值地图和约束地图（统称为价值地图），这些地图以机器人观察空间为基础（a）。然后，组合后的价值地图作为运动规划器的目标函数，用于合成机器人的操作轨迹（b）。整个过程不涉及任何额外的训练。[文献链接](https://arxiv.org/abs/2307.05973)&[项目链接](https://voxposer.github.io/)*


## 文献四
<!-- Using HTML to center the abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>    </h2>
        <h2>VoxAct-B: Voxel-Based Acting and Stabilizing Policy for Bimanual Manipulation</h2>
        <h2 style="font-size: 16px;">作者：I-Chun Arthur Liu, Sicheng He, Daniel Seita*, Gaurav S. Sukhatme*</h2>
        <h2 style="font-size: 16px;">单位：University of Southern California</h2>
        <h2 style="font-size: 16px;">Conference on Robot Learning (CoRL) 2024</h2>
        <div class="content has-text-justified">
摘要：双手操作对许多机器人应用至关重要。与单臂操作相比，由于更高维度的动作空间，双手操作任务具有挑战性。先前的工作利用大量数据和原始操作来解决这个问题，但可能会受到样本效率低下和各种任务泛化能力有限的影响。为此，我们提出了VoxAct-B，这是一种基于语言条件的体素方法，利用视觉语言模型（VLM）对场景中的关键区域进行优先级排序，并重建体素网格。我们将此体素网格提供给我们的双手操作策略，以学习动作和稳定动作。这种方法能够从体素中更有效地学习策略，并可推广到不同的任务。在仿真中，我们表明VoxAct-B在细粒度双手操作任务上优于强基线。此外，我们使用两个UR5在现实世界的Open Drawer和Open Jar任务中演示了VoxAct-B。代码、数据和视频可在https://voxact-b.github.io.
        </div>
    </div>
</div>

---
![Turing Machine](/static/image/4.jpeg)

*图 4: ·VoxAct-B使用体素表示和语言从双臂进行6-DoF操作的双手操作。我们在模拟中测试了四个语言条件的双手任务，并在两个UR5的真实设置中测试了两个（打开抽屉和打开罐子）。[文献链接](https://arxiv.org/abs/2407.04152)&[项目链接](https://voxact-b.github.io/)*


## 文献五
<!-- Using HTML to center the abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>    </h2>
        <h2>LaMI: Large Language Models for Multi-Modal Human-Robot Interaction</h2>
        <h2 style="font-size: 16px;">作者：Chao Wang, Stephan Hasler, Daniel Tanneberg, Felix Ocker, Frank Joublin, Antonello Ceravola, Joerg Deigmoeller, Michael Gienger</h2>
        <h2 style="font-size: 16px;">单位：Honda Research Institute EU Ofenbach am Main, Germany</h2>
        <h2 style="font-size: 16px;">Extended Abstracts of the CHI Conference on Human Factors in Computing Systems. 2024: 1-10.</h2>
        <div class="content has-text-justified">
摘要：本文提出了一种基于大型语言模型（LLM）的创新机器人系统，用于增强多模态人机交互（HRI）。传统的HRI系统依赖于复杂的设计来进行意图估计、推理和行为生成，这些都是资源密集型的。相比之下，我们的系统使研究人员和从业者能够通过三个关键方面来规范机器人行为：提供高级语言指导，创建机器人可以使用的“原子动作”和表达，以及提供一组示例。在物理机器人上实现，它演示了熟练适应多模式输入，并根据研究人员制定的指导方针确定适当的行动方式来帮助人类使用手臂。同时，它通过语音输出协调机器人的盖子、脖子和耳朵的运动，以产生动态的多模态表情。这展示了该系统通过从传统的手动状态和流程设计方法转变为直观、基于指导和示例驱动的方法来彻底改变HRI的潜力。补充材料可在以下网址找到https://hri-eu.github.io/Lami/
        </div>
    </div>
</div>

---
![Turing Machine](/static/image/5LaMI.jpeg)

*图 5: 以制导、能力和示例为中心的LLM驱动的人机交互 [文献链接](https://arxiv.org/abs/2401.15174)&[项目链接](https://hri-eu.github.io/Lami/)*



## 文献六
<!-- Using HTML to center the abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>    </h2>
        <h2>ManiFoundation Model for General-Purpose Robotic Manipulation of
 Contact Synthesis with Arbitrary Objects and Robots</h2>
        <h2 style="font-size: 16px;">作者：Zhixuan Xu1∗
, Chongkai Gao1∗
, Zixuan Liu2∗
, Gang Yang1∗
, Chenrui Tie3
, Haozhuo Zheng4
, Haoyu Zhou5
,
Weikun Peng1
, Debang Wang1
, Tianrun Hu1
, Tianyi Chen6
, Zhouliang Yu7
, Lin Shao1†
</h2>
        <h2 style="font-size: 16px;">单位：<br>
1Department of Computer Science, National University of Singapore,  
2Tsinghua University,  
3Peking University,  
4Department of Mathematics, National University of Singapore,  
5Department of Mechanical Engineering, National University of Singapore, 
6Shanghai Jiao Tong University 7Hongkong University of Science and Technology</h2>
        <h2 style="font-size: 16px;">Accepted to IROS 2024 (Oral Presentation)</h2>
        <div class="content has-text-justified">
摘要：为了大幅提高机器人智能，迫切需要开发一个大型模型，使通用机器人能够熟练地执行各种操作任务，类似于LLM所表现出的多功能任务规划能力。物体、机器人和操纵任务的巨大多样性带来了巨大的挑战。我们的工作引入了一个全面的框架来开发通用机器人操纵的基础模型，该模型将操纵任务形式化为接触合成。具体来说，我们的模型将对象和机器人操纵器点云、对象物理属性、目标运动和操纵区域掩码作为输入。它输出物体上的接触点以及相关的接触力或接触后运动，以便机器人实现所需的操纵任务。我们在模拟和现实环境中进行了广泛的实验，操纵铰接的刚性物体、刚性物体和维度不同的可变形物体，从绳索等一维物体到布料等二维物体，再到橡皮泥等三维物体。我们的模型实现了约90%的平均成功率。补充材料和视频可在我们的项目网站上找到，网址为 https://manifoundationmodel.github.io/.
        </div>
    </div>
</div>

---
![Turing Machine](/static/image/6ManiFoundation.jpeg)

*图 6: 我们ManiFoundation模型的管道。左：我们将操纵任务分解为基于VLM的规划或流模型中的一系列对象点云运动。中间：我们训练一个ManiFoundation网络来预测序列中每个运动的接触点和力热图。右：我们根据接触点和力热图的初始结果进行优化，获取机器人的执行姿态。[文献链接](https://arxiv.org/abs/2405.06964)&[项目链接](https://manifoundationmodel.github.io/)*


## 文献七
<!-- Using HTML to center the abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>    </h2>
        <h2> ManipLLM: Embodied Multimodal Large Language Model for
 Object-Centric Robotic Manipulation</h2>
        <h2 style="font-size: 16px;">作者：Xiaoqi Li, Mingxu Zhang, Yiran Geng, Haoran Geng, Yuxing Long, Yan Shen, Renrui Zhang, Jiaming Liu, Hao Dong</h2>
        <h2 style="font-size: 16px;">单位：1School of Computer Science, Peking University
2 Beijing University of Posts and Telecommunications 3 MMLab, CUHK</h2>
        <h2 style="font-size: 16px;">Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 18061-18070</h2>
        <div class="content has-text-justified">
摘要: 机器人操纵依赖于准确预测接触点和末端执行器方向，以确保成功操作。然而，基于学习的机器人操作，在模拟器内的有限类别上训练，往往难以实现通用性，尤其是在面对广泛类别时。因此，我们引入了一种创新的机器人操纵方法，该方法利用多模态大语言模型（MLLM）的鲁棒推理能力来提高操纵的稳定性和泛化能力。通过微调注入的适配器，我们保留了MLLM固有的常识和推理能力，同时为它们配备了操作能力。基本见解在于引入了微调范式，包括对象类别理解、启示先验推理和以对象为中心的姿势预测，以激发MLLM在操纵中的推理能力。在推理过程中，我们的方法利用RGB图像和文本提示来预测思维链中末端执行器的姿势。在建立初始联系后，引入主动阻抗自适应策略，以闭环方式规划即将到来的航路点。此外，在现实世界中，我们设计了一种测试时间自适应（TTA）策略进行操作，使模型能够更好地适应当前的现实世界场景配置。模拟器和现实世界的实验表明了ManipLLM的良好性能。更多详细信息和演示可以在https://sites.google.com/view/manipllm.
        </div>
    </div>
</div>

---
![Turing Machine](/static/image/7ManipLLM.jpeg)

*图 7: ManipLLM的训练细节。该范式包含四个训练任务，使模型能够识别当前对象（类别级别），了解可以操纵哪些区域（区域级别），并最终生成精确的末端执行器姿势（姿势级别）。[文献链接](https://arxiv.org/abs/2312.16217)&[项目链接](https://sites.google.com/view/manipllm)*


## 文献八
<!-- Using HTML to center the abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>    </h2>
        <h2>Physically Grounded Vision-Language Models for Robotic Manipulation</h2>
        <h2 style="font-size: 16px;">作者：Jensen Gao1, Bidipta Sarkar1, Fei Xia2, Ted Xiao2, Jiajun Wu1,
Brian Ichter2, Anirudha Majumdar2,3, Dorsa Sadigh1,2</h2>
        <h2 style="font-size: 16px;">单位：1Stanford University, 2Google DeepMind, 3Princeton University</h2>
        <h2 style="font-size: 16px;">2024 IEEE International Conference on Robotics and Automation (ICRA)</h2>
        <div class="content has-text-justified">
摘要：视觉语言模型（VLMs）的最新进展提高了视觉问答和图像字幕等任务的性能。因此，这些模型现在能够很好地推理物理世界，特别是在机器人操纵等领域。然而，目前的VLM对常见物体的物理概念（如材料、脆弱性）的理解有限，这限制了它们在涉及此类物体的交互和物理推理的机器人操纵任务中的有用性。为了解决这一局限性，我们提出了PHYSOBJECTS，这是一个以对象为中心的数据集，包含39.6K个众包和417K个常见家用物品的自动物理概念注释。我们证明，通过从视觉外观中捕捉这些概念的人类先验，在PHYSOBJECTS上微调VLM可以提高其对物理对象概念的理解，包括对所提出概念的泛化。我们将这种基于物理的VLM整合到一个具有大型语言模型的机器人规划器的交互式框架中，并与不利用基于物理的VLS的基线相比，在需要对物理对象概念进行推理的任务上显示出改进的规划性能。我们还说明了我们的物理接地VLM在真实机器人上的好处，它提高了任务成功率。我们发布了我们的数据集，并在https://iliad.stanford.edu/pg-vlm/。
        </div>
    </div>
</div>

---
![Turing Machine](/static/image/8Physically.jpeg)

*图 8: （a） 我们收集常见家居物品的物理概念注释，以微调VLM。（b） 我们在基于LLM的机器人规划框架中使用微调的VLM，其中LLM在生成计划之前向VLM查询场景中对象的物理概念。（c） 我们在真正的Franka Emika Panda机器人上评估LLM生成的计划。[文献链接](https://arxiv.org/abs/2309.02561)&[项目链接](https://iliad.stanford.edu/pg-vlm/)*



## 文献九
<!-- Using HTML to center the abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>    </h2>
        <h2>Instruct2Act: Mapping Multi-modality Instructions
 to Robotic Actions with Large Language Model</h2>
        <h2 style="font-size: 16px;">作者：Siyuan Huang1,2 Zhengkai Jiang4 Hao Dong3 Yu Qiao2 Peng Gao2 Hongsheng Li5</h2>
        <h2 style="font-size: 16px;">单位：1 Shanghai Jiao Tong University, 2 Shanghai AI Laboratory, 3 CFCS, School of CS, PKU
4 University of Chinese Academy of Sciences, 5 The Chinese University of Hong Kong</h2>
        <h2 style="font-size: 16px;">Huang S, Jiang Z, Dong H, et al. Instruct2act: Mapping multi-modality instructions to robotic actions with large language model[J]. arXiv preprint arXiv:2305.11176, 2023.</h2>
        <div class="content has-text-justified">
摘要：基础模型在各种应用中取得了重大进展，包括文本到图像生成、全景分割和自然语言处理。本文介绍了Instruct2Act，这是一个利用大型语言模型将多模态指令映射到机器人操纵任务的顺序动作的框架。具体来说，Instruct2Act使用LLM模型生成Python程序，这些程序构成了机器人任务的全面感知、规划和动作循环。在感知部分，预定义的API用于访问多个基础模型，其中分段任意模型（SAM）准确地定位候选对象，CLIP对其进行分类。通过这种方式，该框架利用基础模型的专业知识和机器人能力，将复杂的高级指令转换为精确的策略代码。我们的方法是可调整和灵活的，可以适应各种教学方式和输入类型，并满足特定的任务需求。我们通过在桌面操作领域的不同场景中对机器人任务进行评估，验证了我们方法的实用性和效率。此外，我们的零样本方法在几个任务中优于许多最先进的基于学习的策略。我们提出的方法的代码可在https://github.com/OpenGVLab/Instruct2Act，作为具有各种模态输入的高级机器人指令任务的稳健基准。
        </div>
    </div>
</div>

---
![Turing Machine](/static/image/9Instruct2Act.jpeg)

*图 9: 我们提出的Instruct2Act框架的范式。从任务指令开始，该框架利用LLM生成可执行代码，通过API调用可视化基础模型来识别环境。利用识别的对象语义信息，我们生成合理的动作，并将其发送到低级控制器以执行任务。绿色和蓝色的指令分别代表纯语言和多模式指令。[文献链接](https://arxiv.org/abs/2305.11176)&[项目链接](https://github.com/OpenGVLab/Instruct2Act)*


## 文献十
<!-- Using HTML to center the abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>    </h2>
        <h2>TidyBot: Personalized Robot Assistance with Large Language Models</h2>
        <h2 style="font-size: 16px;">作者：Jimmy Wu1 · Rika Antonova2 · Adam Kan3 · Marion Lepert2 · Andy Zeng4 · Shuran Song5 · Jeannette Bohg2 ·
Szymon Rusinkiewicz1 · Thomas Funkhouser1,4</h2>
        <h2 style="font-size: 16px;">单位：1 Princeton University, Princeton, NJ, USA
2 Stanford University, Stanford, CA, USA
3 The Nueva School, San Mateo, CA, USA
4 Google, Mountain View, CA, USA
5 Columbia University, New York, NY, USA</h2>
        <h2 style="font-size: 16px;">Autonomous Robots (2023) 47:1087–1102</h2>
        <div class="content has-text-justified">
摘要：为了使机器人有效地个性化物理辅助，它必须学习用户偏好，这些偏好通常可以重新应用于未来的场景。在这项工作中，我们研究了使用机器人进行家庭清洁的个性化，这些机器人可以通过捡起物体并将其放好来整理房间。一个关键的挑战是确定放置每个物品的合适位置，因为人们的偏好可能因个人品味或文化背景而异。例如，一个人可能更喜欢把衬衫放在抽屉里，而另一个人可能喜欢把它们放在架子上。我们的目标是构建一个系统，通过与特定人的预先互动，从少数例子中学习这些偏好。我们表明，机器人可以将基于语言的规划和感知与大型语言模型（LLM）的少镜头摘要功能相结合，以推断出广泛适用于未来交互的广义用户偏好。这种方法能够快速适应，并在我们的基准数据集中对看不见的对象达到91.2%的准确率。我们还在一个名为TidyBot的真实世界移动操纵器上演示了我们的方法，该操纵器在真实世界的测试场景中成功地放置了85.0%的物体。
        </div>
    </div>
</div>

---
![Turing Machine](/static/image/10TidyBot.jpeg)

*图 10: 系统概述：一旦用LLM总结了用户的偏好，TidyBot将定位地板上最近的物体，用其以自我为中心的相机移动以获得特写视图，使用CLIP预测物体的类别，使用LLM总结的规则选择容器和操纵图元，然后执行图元将物体放入所选的容器中，重复整个过程，直到地板上找不到更多的物体。[文献链接](https://arxiv.org/abs/2305.05658)*


## 文献十一
<!-- Using HTML to center the abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>    </h2>
        <h2>GraspGPT: Leveraging Semantic Knowledge from
 a Large Language Model for Task-Oriented Grasping</h2>
        <h2 style="font-size: 16px;">作者：Chao Tang , Dehao Huang , Wenqi Ge , Weiyu Liu , and Hong Zhang</h2>
        <h2 style="font-size: 16px;">单位：Southern University of Science and Technology</h2>
        <h2 style="font-size: 16px;">IEEE Robotics and Automation Letters, vol. 8, no. 11, pp. 7551-7558, 2023.</h2>
        <div class="content has-text-justified">
摘要：面向任务的抓取（TOG）是指预测对对象的抓取，以实现后续操作任务的问题。为了对对象、任务和抓取之间的复杂关系进行建模，现有的方法将语义知识作为先验整合到TOG管道中。然而，现有的语义知识通常是基于封闭世界概念集构建的，这限制了对预定义集合之外的新概念的泛化。为了解决这个问题，我们提出了GraspGPT，这是一个基于大型语言模型（LLM）的TOG框架，它利用LLM的开放式语义知识来实现对新概念的零样本泛化。我们在语言增强任务抓取（LA TaskGrasp）数据集上进行了实验，并证明了当从训练集中推广到新概念时，在不同的保留设置下，抓取GPT的性能优于现有的TOG方法。在真实的机器人实验中进一步验证了GraspGPT的有效性。我们的代码、数据、附录和视频可在以下网址公开获取https://sites.google.com/view/graspgpt.
        </div>
    </div>
</div>

---
![Turing Machine](/static/image/11GraspGPT.jpeg)

*图 11: （a） GrapGPT框架概述：当在自然语言教学中出现一个新概念，如一个新的对象类或任务时，GrapGPT首先提示LLM获取该概念的一组语言描述段落。随后，GrasGPT根据传感器和LLM的多模态输入评估抓取候选者的任务兼容性。（b） 面向任务的掌握评估器的详细结构：该模块是一个定制的转换器解码器，将LLM的语义知识注入自然语言教学中。[文献链接](https://arxiv.org/abs/2307.13204)&[项目链接](https://sites.google.com/view/graspgpt)*




<!-- > Note: This is an example of a Jekyll-based project website template: [Github link](https://github.com/shunzh/project_website).\
> The following content is generated by ChatGPT. The figure is manually added. -->


<!-- ## Youtube Video
<html>
	<head>
		<title>zwh</title>
		<meta charset="utf-8"/>	
	</head>
<body>
	<div>
		<video src="https://eliahuhorwitz.github.io/Academic-project-page-template/static/videos/banner_video.mp4" width="600" height="600" type="video/mp4" controls="controls" autoplay="autoplay" >	</video>
        
	</div>
</body>
</html> -->