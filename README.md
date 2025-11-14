
<div id="sayuri-art" align="center">
    </br>
    <img src="./img/sayuri-art.PNG" alt="Sayuri Art" width="768"/>
    <h3>Sayuri</h3>
</div>

## TensorRT backend test results

~~~
$ bash simple.sh

./sayuri --mode selfplay --config configs/selfplay-config.txt -g 0 --target-directory selfplay --weights-dir weights

Network Version: 5
Input Channels: 43
Residual Blocks: 6
Residual Channels: 96
(Residual Block)
  block 1: ResidualBlock
  block 2: ResidualBlock
  block 3: ResidualBlock-SE
  block 4: ResidualBlock
  block 5: ResidualBlock
  block 6: ResidualBlock-SE
(Nested Bottleneck Block)
  block 1: NestedBottleneckBlock
  block 2: NestedBottleneckBlock
  block 3: NestedBottleneckBlock-SE
  block 4: NestedBottleneckBlock
  block 5: NestedBottleneckBlock
  block 6: NestedBottleneckBlock-SE
(Residual + Mixer Block)
  block 1: ResidualBlock
  block 2: ResidualBlock
  block 3: ResidualBlock-SE
  block 4: ResidualBlock
  block 5: ResidualBlock
  block 6: ResidualBlock
  block 7: MixerBlock-SE-SE
Policy Head Type: RepLK
Policy Head Channels: 24
Value Head Channels: 24
Done! Load the weights file in 0.03 sec.
CUDA version: Major 13, Minor 0
Use cuDNN: No
Number of CUDA devices: 1
=== Device: 0 ===
  Name: NVIDIA GeForce RTX 5060
  Compute capability: 12.0
  Enable the FP16
  Enable the tensor cores
Done constructing network.
Allocated 400.00 MiB memory for NN cache (141509 entries).
============================================
Hash value: 31EC9DC9306CCCBC
Number of parallel games: 64
Target self-play games: 5000
Directory for saving: selfplay

                cuda backend   TensorRT backend  Nested bottleneck  residual+mixer
                       (best)          (best)           (best)           (best)
steps            200K    172K    200K    168K     200K    220K     200K    164K
samples        51200K  44032K  51200K  43008K   51200K  56320K   51200K  41984K 
games            250K    215K    250K    210K     250K    275K     250K    205K
all            4.9080  4.8768  4.8171  4.7905   4.8537  4.8446   4.8872  4.8677
prob           1.9292  1.9324  1.8882  1.9130   1.8775  1.8617   1.8916  1.9164
aux prob       0.4362  0.4337  0.4308  0.4301   0.4295  0.4290   0.4317  0.4307
soft prob      0.5047  0.5044  0.5018  0.5016   0.5012  0.5014   0.5019  0.5021
soft aux prob  0.0791  0.0790  0.0785  0.0783   0.0784  0.0785   0.0786  0.0784
optimistic     0.2675  0.2689  0.2592  0.2616   0.2587  0.2544   0.2582  0.2656
ownership      0.7263  0.7049  0.7208  0.6887   0.7525  0.7665   0.7745  0.7473
wdl            0.6096  0.6017  0.5845  0.5763   0.5850  0.5828   0.5841  0.5675
Q value        0.1676  0.1637  0.1593  0.1550   0.1594  0.1589   0.1584  0.1544
scores         0.1777  0.1783  0.1858  0.1781   0.2025  0.2028   0.2005  0.1980
errors         0.0102  0.0097  0.0082  0.0078   0.0090  0.0085   0.0076  0.0075
time           121.4(h)        54.0(h)          64.2(h) 70.2(h)  60.4(h) 49.6(h)
~~~

![all loss](./img/trt_cuda_loss.png)
![wdl loss](./img/trt_cuda_loss_wdl.png)
![run time](./img/trt_cuda_loss_time.png)
![all0 loss](./img/all_loss.png)
![wdl0 loss](./img/wdl_loss.png)

## Let's ROCK!

**Sayuri** is a GTP-compliant Go engine built on Deep Convolutional Neural Networks and Monte Carlo Tree Search. It learns to play Go from scratch using an AlphaZero-style algorithm, without any handcrafted human strategies. Inspired heavily by **Leela Zero** and **KataGo**, Sayuri initially borrowed its board data structures, search algorithms, and network format from Leela Zero. In later versions, the engine follows KataGo's research and now supports variable rulesets, komi settings, and board sizes.

For development insights and reports, see:
* [Development Log (in Chinese)](https://hackmd.io/@yrHb-fKBRoyrKDEKdPSDWg/BJgfay0Yc)
* [Performance Report before UEC15 (v0.6)](https://drive.google.com/file/d/1ATd_u-E-OnviczsDH8wVL0c3Q1NzUCKW/view?usp=share_link)


## Quick Start via Terminal

To run the engine, you need a executable weights first. The released weights can be got from this [page](./docs/MODEL.md). Then launching the engine with GTP mode via the terminal/PowerShell, using 1 thread and 400 visits per move with optimistic policy. Please type

    $ ./sayuri -w <weights file> -t 1 -p 400 --use-optimistic-policy


After executing the command, you'll see diagnostic output. If this output includes ```Network Version```, it indicates that the engine is successfully running in GPT mode. However, since GPT mode isn't designed for human interaction, you should use the graphical interface (GUI) instead. Please refer to the **Graphical Interface** section for more details.

For a list of additional command-line arguments, use the --help option. Please type:

    $ ./sayuri --help

The default engine uses a Chinese-like rule, which has a tendency to keep playing to remove some dead stones, even when their ownership of an area is clear. This can lead to unwanted capturing moves. To prevent these unnecessary moves, you have two options. First, while using the Chinese-like rule, add the ```--friendly-pass``` option. Second, switch to a Japanese-like rule by using the ```--scoring-rule territory``` option.

You can utilize the pure Python engine with a checkpoint model. The released checkpoint models could be found from this [page](./docs/MODEL.md). Although the Python engine is significantly weaker than the C++ engine, it makes running the raw model much easier. More detail you may see [here](./train/README.md).

    $ python3 train/torch/pysayuri.py -c model.pt --use-swa

## Execute Engine via Graphical Interface

Sayuri is not complete engine. You need a graphical interface for playing with her. She supports any GTP (version 2) interface application. [Sabaki](https://sabaki.yichuanshen.de/) and [GoGui](https://github.com/Remi-Coulom/gogui) are recommended because Sayuri supports some specific analysis commands.

* Sabaki analysis mode

![sabaki-sample01](./img/sabaki-sample01.png)

* GoGui analysis commands

![gogui-sample01](./img/gogui-sample01.png)

## Build From Source

For instructions on building from source, please refer to this [section](./docs/COMPILE.md). If you are using Windows, you can download a precompiled executable directly from the release page.

## Reinforcement Learning

Sayuri is a highly efficient self-play learning system for the game of Go that focuses on computational efficiency. In her v0.7 release, Sayuri’s training cost (represented by the purple line) is notably lower than that of both KataGo and Leela Zero. Compared to ELF OpenGo, Sayuri requires approximately 250× less computation. The complete training run was conducted in three months using a single RTX 4080 GPU. By comparison, KataGo’s g104 version reports a reduction of around 50×, making Sayuri’s efficiency improvement considerably larger.

For details on how to run the self-play loop, please refer to this [guide](./bash/README.md).

![sayuri-vs-kata](./img/sayurivskata-v7.png)

## Other Resources

* Go Text Protocol, [https://www.gnu.org/software/gnugo/gnugo_19.html](https://www.gnu.org/software/gnugo/gnugo_19.html)
* Leela Zero, [https://github.com/leela-zero/leela-zero](https://github.com/leela-zero/leela-zero)
* KataGo methods, [https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)
* [YouTube](https://www.youtube.com/watch?v=82UclNrXGxg), playing with Pachi.
* Supported analysis commands, [analyze](./docs/ANALYZE.md).
* [AlphaZero 之加速演算法實作 (v0.4~v0.5)](https://hackmd.io/@yrHb-fKBRoyrKDEKdPSDWg/HJI9_p70i), describe some methods for old version.

## License

The code is released under the GPLv3, except for threadpool.h, cppattributes.h, Eigen and Fast Float, which have specific licenses mentioned in those files.

## Contact

cglemon000@gmail.com (Hung-Tse Lin)
