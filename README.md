# ***Disentangled Latent Speech Representation for Automatic Pathological Intelligibility Assessment***

![Here should be an image visible.](schematic_digital_v2.png)

## **Introduction**

This is the code repository for the paper ***Disentangled Latent Speech Representation for Automatic Pathological
Intelligibility Assessment***, which can be found [here](https://arxiv.org/abs/2204.04016).

It uses an existing architecture, originally desined with Voice Conversion in mind ([SpeechSplit](https://arxiv.org/abs/2004.11284)). However, for this work it is trained differently from the original implementation and only the encoder outputs are used to extract certain latent speech representations. These can then be used in a final step to determine a (pathological) speaker's intelligibility value based on a reference signal.

If you use this repository for your work, please cite the following paper:

```
@article{weise2022disentangled,
  title = {Disentangled Latent Speech Representation for Automatic Pathological Intelligibility Assessment},
  author = {Weise, Tobias and Klumpp, Philipp and Maier, Andreas and Noeth, Elmar and Heismann, Bjoern and Schuster, Maria and Yang, Seung Hee},
  journal = {arXiv preprint arXiv:2204.04016}
  url = {https://arxiv.org/abs/2204.04016},
  year = {2022},
}
```

## **Requirements**
