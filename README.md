# Slope Disparity Gating: System and Applications

### [Paper](https://ieeexplore.ieee.org/document/9743319)

This repository contains the code implementation and accompanying materials for the paper titled "Slope Disparity Gating: System and Applications" published in the Transactions on Computational Imaging (TCI). The paper introduces a novel active illumination system called Slope Disparity Gating (SDG) and demonstrates its applications in various fields. This system is used to present a new decomposition of the light transport matrix in a scene as a function of disparity.  This README file provides an overview of the repository and instructions on how to use the code.

<br><br>
[Slope Disparity Gating: System and Applications]  
 [Sreenithy Chandran](https://scholar.google.com/citations?user=fab-KeoAAAAJ&hl=en)\*<sup>1</sup>,
 [Hiroyuki Kubo](https://scholar.google.com/citations?user=TRmARjkAAAAJ&hl=en)\*<sup>2</sup>,
[ Tomoki Ueda](https://ieeexplore.ieee.org/author/37086871844)<sup>2</sup> 
 [Takuya Funatomi](https://scholar.google.com/citations?user=OjA6YhgAAAAJ&hl=en)\*<sup>2</sup>,
 [Yasuhiro Mukaigawa](https://scholar.google.com/citations?user=vWuuPa8AAAAJ&hl=en)<sup>2</sup>,
 [Suren Jayasuriya](https://scholar.google.com/citations?user=DEfu2GoAAAAJ&hl=en)<sup>1</sup>,

 <br>
 <sup>1</sup>Arizona State University, <sup>2</sup>Nara Institute of Science and Technology

<!-- Implementation for acquiring light transport of a scene using structured light.
Also shared is the code for a simple Pbrt based renderer for simulating data similar to 

Details about the files


1. lightransport.py-LT acquisition by using the dot scanning projector method

2. video2frame.py- Convert the hadamard projection video file to frames

3. Matlab- This folder contains the files needed for Hadamard code generation based on the size of projector and camera

4. decoding.py- Contains details on how to obtain the light transport matrix using the captured images and perform a simple relighting task
-->