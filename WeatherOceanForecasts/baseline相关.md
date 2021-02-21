“AI Earth”人工智能创新挑战赛Baseline

  

赛题背景

![](https://mmbiz.qpic.cn/mmbiz_png/US10Gcd0tQEfcffueY0reDaT8agHibMbkl6VPJicIaSLBOMT46hKst5wjTztibed2dJsrke6B0nRpRPvJXnC2mlSg/640?wx_fmt=png)

  

![](https://mmbiz.qpic.cn/mmbiz_svg/LwcbhAmMnZBibnjHfa5Dbkk7tP04JCicCSQTxgyGQTicyGIU7M0P5BVzNibgJOO9BU7K91hTpQDPkc1eFEZiaXlGhuoBYTGTn3vc6/640?wx_fmt=svg)

01

赛题简介

发生在热带太平洋上的厄尔尼诺-南方涛动\(ENSO\)现象是地球上最强、最显著的年际气候信号。通过大气或海洋遥相关过程，经常会引发洪涝、干旱、高温、雪灾等极端事件，对全球的天气、气候以及粮食产量具有重要的影响。准确预测ENSO，是提高东亚和全球气候预测水平和防灾减灾的关键。本次赛题是一个时间序列预测问题。基于历史气候观测和模式模拟数据，利用T时刻过去12个月\(包含T时刻\)的时空序列（气象因子），构建预测ENSO的深度学习模型，预测未来1-24个月的Nino3.4指数，如下图所示：

![](https://mmbiz.qpic.cn/mmbiz_jpg/ZQhHsg2x8ficqo4ibQMBLLlIDzHtDotd8aQ4nTGTxuNQeAicZBa9KPgqT95tsd0shdwQVdQEQJg4AXWvM642G6Pug/640?wx_fmt=jpeg)

![](https://mmbiz.qpic.cn/mmbiz_svg/LwcbhAmMnZBibnjHfa5Dbkk7tP04JCicCSQTxgyGQTicyGIU7M0P5BVzNibgJOO9BU7K91hTpQDPkc1eFEZiaXlGhuoBYTGTn3vc6/640?wx_fmt=svg)

02

背景数据描述

### 1\. 数据简介

本次比赛使用的数据包括CMIP5/6模式的历史模拟数据和美国SODA模式重建的近100多年历史观测同化数据。每个样本包含以下气象及时空变量：海表温度异常\(SST\)，热含量异常\(T300\)，纬向风异常（Ua），经向风异常（Va），数据维度为（year,month,lat,lon）。对于训练数据提供对应月份的Nino3.4 index标签数据。

### 2\. 训练数据说明

每个数据样本第一维度（year）表征数据所对应起始年份，对于CMIP数据共4645年，其中1-2265为CMIP6中15个模式提供的151年的历史模拟数据（总共：151年 \*15 个模式=2265）；2266-4645为CMIP5中17个模式提供的140年的历史模拟数据（总共：140年 \*17 个模式=2380）。对于历史观测同化数据为美国提供的SODA数据。其中每个样本第二维度（mouth）表征数据对应的月份，对于训练数据均为36，对应的从当前年份开始连续三年数据（从1月开始，共36月），比如：SODA\_train.nc中\[0,0:36,:,:\]为第1-第3年逐月的历史观测数据；SODA\_train.nc中\[1,0:36,:,:\]为第2-第4年逐月的历史观测数据；…, SODA\_train.nc中\[99,0:36,:,:\]为第100-102年逐月的历史观测数据。和 CMIP\_train.nc中\[0,0:36,:,:\]为CMIP6第一个模式提供的第1-第3年逐月的历史模拟数据；…, CMIP\_train.nc中\[150,0:36,:,:\]为CMIP6第一个模式提供的第151-第153年逐月的历史模拟数据；CMIP\_train.nc中\[151,0:36,:,:\]为CMIP6第二个模式提供的第1-第3年逐月的历史模拟数据；…, CMIP\_train.nc中\[2265,0:36,:,:\]为CMIP5第一个模式提供的第1-第3年逐月的历史模拟数据；…, CMIP\_train.nc中\[2405,0:36,:,:\]为CMIP5第二个模式提供的第1-第3年逐月的历史模拟数据；…, CMIP\_train.nc中\[4644,0:36,:,:\]为CMIP5第17个模式提供的第140-第142年逐月的历史模拟数据。其中每个样本第三、第四维度分别代表经纬度（南纬55度北纬60度，东经0360度），所有数据的经纬度范围相同。

### 3\. 训练数据标签说明

标签数据为Nino3.4 SST异常指数，数据维度为（year,month）。CMIP\(SODA\)\_train.nc对应的标签数据当前时刻Nino3.4 SST异常指数的三个月滑动平均值，因此数据维度与维度介绍同训练数据一致注：三个月滑动平均值为当前月与未来两个月的平均值。

### 4\. 测试数据说明

测试用的初始场（输入）数据为国际多个海洋资料同化结果提供的随机抽取的n段12个时间序列，数据格式采用NPY格式保存，维度为（12，lat，lon, 4）,12为t时刻及过去11个时刻，4为预测因子，并按照SST,T300,Ua,Va的顺序存放。测试集文件序列的命名规则：test\_编号\_起始月份\_终止月份.npy，如test\_00001\_01\_12\_.npy。

![](https://mmbiz.qpic.cn/mmbiz_svg/LwcbhAmMnZBibnjHfa5Dbkk7tP04JCicCSQTxgyGQTicyGIU7M0P5BVzNibgJOO9BU7K91hTpQDPkc1eFEZiaXlGhuoBYTGTn3vc6/640?wx_fmt=svg)

03

评估指标  

  
评分细则说明：根据所提供的n个测试数据，对模型进行测试，得到n组未来1-24个月的序列选取对应预测时效的n个数据与标签值进行计算相关系数和均方根误差，如下图所示。并计算得分。![](https://mmbiz.qpic.cn/mmbiz_jpg/ZQhHsg2x8ficqo4ibQMBLLlIDzHtDotd8aXaGlo3zrPU7Hq6XfcVNrrwXdsPQsg9k3HoCojhezjAI5EAWWJUZeFQ/640?wx_fmt=jpeg)计算公式为:

$$
Score = \frac{2}{3} * accskill - RMSE
$$

其中，

$$
accskill = \sum_{i=1}^{24} a * ln(i) * cor_i, \\
(i \le,a = 1.5; 5 \le i \le 11, a= 2; 12 \le i \le 18,a=3;19 \le i, a = 4)
$$

而：

$$
cor = \frac{\sum(X-\bar(X))\sum(Y-\bar(Y)}{\sqrt{\sum(X-\bar{X})^2)\sum(Y-\bar{Y})^2)}}
$$

<svg xmlns="http://www.w3.org/2000/svg" role="img" focusable="false" viewBox="0 -1728.7 8999.2 2974.6" aria-hidden="true" style="-webkit-overflow-scrolling: touch;vertical-align: -2.819ex;width: 20.36ex;height: 6.73ex;max-width: 300% !important;"><g stroke="currentColor" fill="currentColor" stroke-width="0" transform="matrix(1 0 0 -1 0 0)"><g data-mml-node="math"><g data-mml-node="mi"><path data-c="52" d="M230 637Q203 637 198 638T193 649Q193 676 204 682Q206 683 378 683Q550 682 564 680Q620 672 658 652T712 606T733 563T739 529Q739 484 710 445T643 385T576 351T538 338L545 333Q612 295 612 223Q612 212 607 162T602 80V71Q602 53 603 43T614 25T640 16Q668 16 686 38T712 85Q717 99 720 102T735 105Q755 105 755 93Q755 75 731 36Q693 -21 641 -21H632Q571 -21 531 4T487 82Q487 109 502 166T517 239Q517 290 474 313Q459 320 449 321T378 323H309L277 193Q244 61 244 59Q244 55 245 54T252 50T269 48T302 46H333Q339 38 339 37T336 19Q332 6 326 0H311Q275 2 180 2Q146 2 117 2T71 2T50 1Q33 1 33 10Q33 12 36 24Q41 43 46 45Q50 46 61 46H67Q94 46 127 49Q141 52 146 61Q149 65 218 339T287 628Q287 635 230 637ZM630 554Q630 586 609 608T523 636Q521 636 500 636T462 637H440Q393 637 386 627Q385 624 352 494T319 361Q319 360 388 360Q466 361 492 367Q556 377 592 426Q608 449 619 486T630 554Z"></path></g><g data-mml-node="mi" transform="translate(759, 0)"><path data-c="4D" d="M289 629Q289 635 232 637Q208 637 201 638T194 648Q194 649 196 659Q197 662 198 666T199 671T201 676T203 679T207 681T212 683T220 683T232 684Q238 684 262 684T307 683Q386 683 398 683T414 678Q415 674 451 396L487 117L510 154Q534 190 574 254T662 394Q837 673 839 675Q840 676 842 678T846 681L852 683H948Q965 683 988 683T1017 684Q1051 684 1051 673Q1051 668 1048 656T1045 643Q1041 637 1008 637Q968 636 957 634T939 623Q936 618 867 340T797 59Q797 55 798 54T805 50T822 48T855 46H886Q892 37 892 35Q892 19 885 5Q880 0 869 0Q864 0 828 1T736 2Q675 2 644 2T609 1Q592 1 592 11Q592 13 594 25Q598 41 602 43T625 46Q652 46 685 49Q699 52 704 61Q706 65 742 207T813 490T848 631L654 322Q458 10 453 5Q451 4 449 3Q444 0 433 0Q418 0 415 7Q413 11 374 317L335 624L267 354Q200 88 200 79Q206 46 272 46H282Q288 41 289 37T286 19Q282 3 278 1Q274 0 267 0Q265 0 255 0T221 1T157 2Q127 2 95 1T58 0Q43 0 39 2T35 11Q35 13 38 25T43 40Q45 46 65 46Q135 46 154 86Q158 92 223 354T289 629Z"></path></g><g data-mml-node="mi" transform="translate(1810, 0)"><path data-c="53" d="M308 24Q367 24 416 76T466 197Q466 260 414 284Q308 311 278 321T236 341Q176 383 176 462Q176 523 208 573T273 648Q302 673 343 688T407 704H418H425Q521 704 564 640Q565 640 577 653T603 682T623 704Q624 704 627 704T632 705Q645 705 645 698T617 577T585 459T569 456Q549 456 549 465Q549 471 550 475Q550 478 551 494T553 520Q553 554 544 579T526 616T501 641Q465 662 419 662Q362 662 313 616T263 510Q263 480 278 458T319 427Q323 425 389 408T456 390Q490 379 522 342T554 242Q554 216 546 186Q541 164 528 137T492 78T426 18T332 -20Q320 -22 298 -22Q199 -22 144 33L134 44L106 13Q83 -14 78 -18T65 -22Q52 -22 52 -14Q52 -11 110 221Q112 227 130 227H143Q149 221 149 216Q149 214 148 207T144 186T142 153Q144 114 160 87T203 47T255 29T308 24Z"></path></g><g data-mml-node="mi" transform="translate(2455, 0)"><path data-c="45" d="M492 213Q472 213 472 226Q472 230 477 250T482 285Q482 316 461 323T364 330H312Q311 328 277 192T243 52Q243 48 254 48T334 46Q428 46 458 48T518 61Q567 77 599 117T670 248Q680 270 683 272Q690 274 698 274Q718 274 718 261Q613 7 608 2Q605 0 322 0H133Q31 0 31 11Q31 13 34 25Q38 41 42 43T65 46Q92 46 125 49Q139 52 144 61Q146 66 215 342T285 622Q285 629 281 629Q273 632 228 634H197Q191 640 191 642T193 659Q197 676 203 680H757Q764 676 764 669Q764 664 751 557T737 447Q735 440 717 440H705Q698 445 698 453L701 476Q704 500 704 528Q704 558 697 578T678 609T643 625T596 632T532 634H485Q397 633 392 631Q388 629 386 622Q385 619 355 499T324 377Q347 376 372 376H398Q464 376 489 391T534 472Q538 488 540 490T557 493Q562 493 565 493T570 492T572 491T574 487T577 483L544 351Q511 218 508 216Q505 213 492 213Z"></path></g><g data-mml-node="mo" transform="translate(3496.8, 0)"><path data-c="3D" d="M56 347Q56 360 70 367H707Q722 359 722 347Q722 336 708 328L390 327H72Q56 332 56 347ZM56 153Q56 168 72 173H708Q722 163 722 153Q722 140 707 133H70Q56 140 56 153Z"></path></g><g data-mml-node="munderover" transform="translate(4552.6, 0)"><g data-mml-node="mo"><path data-c="2211" d="M60 948Q63 950 665 950H1267L1325 815Q1384 677 1388 669H1348L1341 683Q1320 724 1285 761Q1235 809 1174 838T1033 881T882 898T699 902H574H543H251L259 891Q722 258 724 252Q725 250 724 246Q721 243 460 -56L196 -356Q196 -357 407 -357Q459 -357 548 -357T676 -358Q812 -358 896 -353T1063 -332T1204 -283T1307 -196Q1328 -170 1348 -124H1388Q1388 -125 1381 -145T1356 -210T1325 -294L1267 -449L666 -450Q64 -450 61 -448Q55 -446 55 -439Q55 -437 57 -433L590 177Q590 178 557 222T452 366T322 544L56 909L55 924Q55 945 60 948Z"></path></g><g data-mml-node="TeXAtom" transform="translate(148.2, -1087.9) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mi"><path data-c="69" d="M184 600Q184 624 203 642T247 661Q265 661 277 649T290 619Q290 596 270 577T226 557Q211 557 198 567T184 600ZM21 287Q21 295 30 318T54 369T98 420T158 442Q197 442 223 419T250 357Q250 340 236 301T196 196T154 83Q149 61 149 51Q149 26 166 26Q175 26 185 29T208 43T235 78T260 137Q263 149 265 151T282 153Q302 153 302 143Q302 135 293 112T268 61T223 11T161 -11Q129 -11 102 10T74 74Q74 91 79 106T122 220Q160 321 166 341T173 380Q173 404 156 404H154Q124 404 99 371T61 287Q60 286 59 284T58 281T56 279T53 278T49 278T41 278H27Q21 284 21 287Z"></path></g><g data-mml-node="mo" transform="translate(345, 0)"><path data-c="3D" d="M56 347Q56 360 70 367H707Q722 359 722 347Q722 336 708 328L390 327H72Q56 332 56 347ZM56 153Q56 168 72 173H708Q722 163 722 153Q722 140 707 133H70Q56 140 56 153Z"></path></g><g data-mml-node="mn" transform="translate(1123, 0)"><path data-c="31" d="M213 578L200 573Q186 568 160 563T102 556H83V602H102Q149 604 189 617T245 641T273 663Q275 666 285 666Q294 666 302 660V361L303 61Q310 54 315 52T339 48T401 46H427V0H416Q395 3 257 3Q121 3 100 0H88V46H114Q136 46 152 46T177 47T193 50T201 52T207 57T213 61V578Z"></path></g></g><g data-mml-node="TeXAtom" transform="translate(368.4, 1150) scale(0.707)" data-mjx-texclass="ORD"><g data-mml-node="mn"><path data-c="32" d="M109 429Q82 429 66 447T50 491Q50 562 103 614T235 666Q326 666 387 610T449 465Q449 422 429 383T381 315T301 241Q265 210 201 149L142 93L218 92Q375 92 385 97Q392 99 409 186V189H449V186Q448 183 436 95T421 3V0H50V19V31Q50 38 56 46T86 81Q115 113 136 137Q145 147 170 174T204 211T233 244T261 278T284 308T305 340T320 369T333 401T340 431T343 464Q343 527 309 573T212 619Q179 619 154 602T119 569T109 550Q109 549 114 549Q132 549 151 535T170 489Q170 464 154 447T109 429Z"></path><path data-c="34" d="M462 0Q444 3 333 3Q217 3 199 0H190V46H221Q241 46 248 46T265 48T279 53T286 61Q287 63 287 115V165H28V211L179 442Q332 674 334 675Q336 677 355 677H373L379 671V211H471V165H379V114Q379 73 379 66T385 54Q393 47 442 46H471V0H462ZM293 211V545L74 212L183 211H293Z" transform="translate(500, 0)"></path></g></g></g><g data-mml-node="mi" transform="translate(6163.2, 0)"><path data-c="72" d="M21 287Q22 290 23 295T28 317T38 348T53 381T73 411T99 433T132 442Q161 442 183 430T214 408T225 388Q227 382 228 382T236 389Q284 441 347 441H350Q398 441 422 400Q430 381 430 363Q430 333 417 315T391 292T366 288Q346 288 334 299T322 328Q322 376 378 392Q356 405 342 405Q286 405 239 331Q229 315 224 298T190 165Q156 25 151 16Q138 -11 108 -11Q95 -11 87 -5T76 7T74 17Q74 30 114 189T154 366Q154 405 128 405Q107 405 92 377T68 316T57 280Q55 278 41 278H27Q21 284 21 287Z"></path></g><g data-mml-node="mi" transform="translate(6614.2, 0)"><path data-c="6D" d="M21 287Q22 293 24 303T36 341T56 388T88 425T132 442T175 435T205 417T221 395T229 376L231 369Q231 367 232 367L243 378Q303 442 384 442Q401 442 415 440T441 433T460 423T475 411T485 398T493 385T497 373T500 364T502 357L510 367Q573 442 659 442Q713 442 746 415T780 336Q780 285 742 178T704 50Q705 36 709 31T724 26Q752 26 776 56T815 138Q818 149 821 151T837 153Q857 153 857 145Q857 144 853 130Q845 101 831 73T785 17T716 -10Q669 -10 648 17T627 73Q627 92 663 193T700 345Q700 404 656 404H651Q565 404 506 303L499 291L466 157Q433 26 428 16Q415 -11 385 -11Q372 -11 364 -4T353 8T350 18Q350 29 384 161L420 307Q423 322 423 345Q423 404 379 404H374Q288 404 229 303L222 291L189 157Q156 26 151 16Q138 -11 108 -11Q95 -11 87 -5T76 7T74 17Q74 30 112 181Q151 335 151 342Q154 357 154 369Q154 405 129 405Q107 405 92 377T69 316T57 280Q55 278 41 278H27Q21 284 21 287Z"></path></g><g data-mml-node="mi" transform="translate(7492.2, 0)"><path data-c="73" d="M131 289Q131 321 147 354T203 415T300 442Q362 442 390 415T419 355Q419 323 402 308T364 292Q351 292 340 300T328 326Q328 342 337 354T354 372T367 378Q368 378 368 379Q368 382 361 388T336 399T297 405Q249 405 227 379T204 326Q204 301 223 291T278 274T330 259Q396 230 396 163Q396 135 385 107T352 51T289 7T195 -10Q118 -10 86 19T53 87Q53 126 74 143T118 160Q133 160 146 151T160 120Q160 94 142 76T111 58Q109 57 108 57T107 55Q108 52 115 47T146 34T201 27Q237 27 263 38T301 66T318 97T323 122Q323 150 302 164T254 181T195 196T148 231Q131 256 131 289Z"></path></g><g data-mml-node="msub" transform="translate(7961.2, 0)"><g data-mml-node="mi"><path data-c="65" d="M39 168Q39 225 58 272T107 350T174 402T244 433T307 442H310Q355 442 388 420T421 355Q421 265 310 237Q261 224 176 223Q139 223 138 221Q138 219 132 186T125 128Q125 81 146 54T209 26T302 45T394 111Q403 121 406 121Q410 121 419 112T429 98T420 82T390 55T344 24T281 -1T205 -11Q126 -11 83 42T39 168ZM373 353Q367 405 305 405Q272 405 244 391T199 357T170 316T154 280T149 261Q149 260 169 260Q282 260 327 284T373 353Z"></path></g><g data-mml-node="mi" transform="translate(466, -150) scale(0.707)"><path data-c="69" d="M184 600Q184 624 203 642T247 661Q265 661 277 649T290 619Q290 596 270 577T226 557Q211 557 198 567T184 600ZM21 287Q21 295 30 318T54 369T98 420T158 442Q197 442 223 419T250 357Q250 340 236 301T196 196T154 83Q149 61 149 51Q149 26 166 26Q175 26 185 29T208 43T235 78T260 137Q263 149 265 151T282 153Q302 153 302 143Q302 135 293 112T268 61T223 11T161 -11Q129 -11 102 10T74 74Q74 91 79 106T122 220Q160 321 166 341T173 380Q173 404 156 404H154Q124 404 99 371T61 287Q60 286 59 284T58 281T56 279T53 278T49 278T41 278H27Q21 284 21 287Z"></path></g></g><g data-mml-node="mo" transform="translate(8721.2, 0)"><path data-c="2C" d="M78 35T78 60T94 103T137 121Q165 121 187 96T210 8Q210 -27 201 -60T180 -117T154 -158T130 -185T117 -194Q113 -194 104 -185T95 -172Q95 -168 106 -156T131 -126T157 -76T173 -3V9L172 8Q170 7 167 6T161 3T152 1T140 0Q113 0 96 17Z"></path></g></g></g></svg>

线下数据转换  

![](https://mmbiz.qpic.cn/mmbiz_png/US10Gcd0tQEfcffueY0reDaT8agHibMbkl6VPJicIaSLBOMT46hKst5wjTztibed2dJsrke6B0nRpRPvJXnC2mlSg/640?wx_fmt=png)

  

 -    将数据转化为我们所熟悉的形式，每个人的风格不一样，此处可以作为如何将nc文件转化为csv等文件

```
## 工具包导入&数据读取
```

### 1\. 数据读取

#### SODA\_label处理

 1.     标签含义

```
标签数据为Nino3.4 SST异常指数，数据维度为（year,month）。  CMIP(SODA)_train.nc对应的标签数据当前时刻Nino3.4 SST异常指数的三个月滑动平均值，因此数据维度与维度介绍同训练数据一致注：三个月滑动平均值为当前月与未来两个月的平均值。
```

 2.     将标签转化为我们熟悉的pandas形式

```
label_path       = './data/SODA_label.nc'label_trans_path = './data/' nc_label         = Dataset(label_path,'r') years            = np.array(nc_label['year'][:])months           = np.array(nc_label['month'][:])year_month_index = []vs               = []for i,year in enumerate(years):    for j,month in enumerate(months):        year_month_index.append('year_{}_month_{}'.format(year,month))        vs.append(np.array(nc_label['nino'][i,j]))df_SODA_label               = pd.DataFrame({'year_month':year_month_index}) df_SODA_label['year_month'] = year_month_indexdf_SODA_label['label']      = vsdf_SODA_label.to_csv(label_trans_path + 'df_SODA_label.csv',index = None)
```

```
df_label.head()
```

|   
 | year\_month | label |
| :-- | :-- | :-- |
| 0 | year\_1\_month\_1 | \-0.40720701217651367 |
| 1 | year\_1\_month\_2 | \-0.20244435966014862 |
| 2 | year\_1\_month\_3 | \-0.10386104136705399 |
| 3 | year\_1\_month\_4 | \-0.02910841442644596 |
| 4 | year\_1\_month\_5 | \-0.13252995908260345 |

### 2\. 数据格式转化

#### 2.1 SODA\_train处理

```
SODA_train.nc中[0,0:36,:,:]为第1-第3年逐月的历史观测数据；SODA_train.nc中[1,0:36,:,:]为第2-第4年逐月的历史观测数据；…,SODA_train.nc中[99,0:36,:,:]为第100-102年逐月的历史观测数据。
```

```
SODA_path        = './data/SODA_train.nc'nc_SODA          = Dataset(SODA_path,'r') 
```

- 自定义抽取对应数据\&转化为df的形式；

> index为年月; columns为lat和lon的组合

```
def trans_df(df, vals, lats, lons, years, months):    '''        (100, 36, 24, 72) -- year, month,lat,lon     '''     for j,lat_ in enumerate(lats):        for i,lon_ in enumerate(lons):            c = 'lat_lon_{}_{}'.format(int(lat_),int(lon_))              v = []            for y in range(len(years)):                for m in range(len(months)):                     v.append(vals[y,m,j,i])            df[c] = v    return df
```

```
year_month_index = []years              = np.array(nc_SODA['year'][:])months             = np.array(nc_SODA['month'][:])lats             = np.array(nc_SODA['lat'][:])lons             = np.array(nc_SODA['lon'][:])for year in years:    for month in months:        year_month_index.append('year_{}_month_{}'.format(year,month))df_sst  = pd.DataFrame({'year_month':year_month_index}) df_t300 = pd.DataFrame({'year_month':year_month_index}) df_ua   = pd.DataFrame({'year_month':year_month_index}) df_va   = pd.DataFrame({'year_month':year_month_index})
```

```
%%timedf_sst = trans_df(df = df_sst, vals = np.array(nc_SODA['sst'][:]), lats = lats, lons = lons, years = years, months = months)df_t300 = trans_df(df = df_t300, vals = np.array(nc_SODA['t300'][:]), lats = lats, lons = lons, years = years, months = months)df_ua   = trans_df(df = df_ua, vals = np.array(nc_SODA['ua'][:]), lats = lats, lons = lons, years = years, months = months)df_va   = trans_df(df = df_va, vals = np.array(nc_SODA['va'][:]), lats = lats, lons = lons, years = years, months = months)
```

```
label_trans_path = './data/'df_sst.to_csv(label_trans_path  + 'df_sst_SODA.csv',index = None)df_t300.to_csv(label_trans_path + 'df_t300_SODA.csv',index = None)df_ua.to_csv(label_trans_path   + 'df_ua_SODA.csv',index = None)df_va.to_csv(label_trans_path   + 'df_va_SODA.csv',index = None)
```

#### 2.2 CMIP\_label处理

```
label_path       = './data/CMIP_label.nc'label_trans_path = './data/'nc_label         = Dataset(label_path,'r') years            = np.array(nc_label['year'][:])months           = np.array(nc_label['month'][:])year_month_index = []vs               = []for i,year in enumerate(years):    for j,month in enumerate(months):        year_month_index.append('year_{}_month_{}'.format(year,month))        vs.append(np.array(nc_label['nino'][i,j]))df_CMIP_label               = pd.DataFrame({'year_month':year_month_index}) df_CMIP_label['year_month'] = year_month_indexdf_CMIP_label['label']      = vsdf_CMIP_label.to_csv(label_trans_path + 'df_CMIP_label.csv',index = None)
```

```
df_CMIP_label.head()
```

|   
 | year\_month | label |
| :-- | :-- | :-- |
| 0 | year\_1\_month\_1 | \-0.26102548837661743 |
| 1 | year\_1\_month\_2 | \-0.1332537680864334 |
| 2 | year\_1\_month\_3 | \-0.014831557869911194 |
| 3 | year\_1\_month\_4 | 0.10506672412157059 |
| 4 | year\_1\_month\_5 | 0.24070978164672852 |

#### 2.3 CMIP\_train处理

```
CMIP_train.nc中[0,0:36,:,:]为CMIP6第一个模式提供的第1-第3年逐月的历史模拟数据；…,CMIP_train.nc中[150,0:36,:,:]为CMIP6第一个模式提供的第151-第153年逐月的历史模拟数据；CMIP_train.nc中[151,0:36,:,:]为CMIP6第二个模式提供的第1-第3年逐月的历史模拟数据；…,CMIP_train.nc中[2265,0:36,:,:]为CMIP5第一个模式提供的第1-第3年逐月的历史模拟数据；…,CMIP_train.nc中[2405,0:36,:,:]为CMIP5第二个模式提供的第1-第3年逐月的历史模拟数据；…,CMIP_train.nc中[4644,0:36,:,:]为CMIP5第17个模式提供的第140-第142年逐月的历史模拟数据。其中每个样本第三、第四维度分别代表经纬度（南纬55度北纬60度，东经0360度），所有数据的经纬度范围相同。
```

```
CMIP_path       = './data/CMIP_train.nc'CMIP_trans_path = './data'nc_CMIP  = Dataset(CMIP_path,'r') 
```

```
nc_CMIP.variables.keys()
```

```
dict_keys(['sst', 't300', 'ua', 'va', 'year', 'month', 'lat', 'lon'])
```

```
nc_CMIP['t300'][:].shape
```

```
(4645, 36, 24, 72)
```

```
year_month_index = []years              = np.array(nc_CMIP['year'][:])months             = np.array(nc_CMIP['month'][:])lats               = np.array(nc_CMIP['lat'][:])lons               = np.array(nc_CMIP['lon'][:])last_thre_years = 1000for year in years:    '''        数据的原因，我们    '''    if year >= 4645 - last_thre_years:        for month in months:            year_month_index.append('year_{}_month_{}'.format(year,month))df_CMIP_sst  = pd.DataFrame({'year_month':year_month_index}) df_CMIP_t300 = pd.DataFrame({'year_month':year_month_index}) df_CMIP_ua   = pd.DataFrame({'year_month':year_month_index}) df_CMIP_va   = pd.DataFrame({'year_month':year_month_index})
```

 -    因为内存限制,我们暂时取最后1000个year的数据

```
def trans_thre_df(df, vals, lats, lons, years, months, last_thre_years = 1000):    '''        (4645, 36, 24, 72) -- year, month,lat,lon     '''     for j,lat_ in (enumerate(lats)):#         print(j)        for i,lon_ in enumerate(lons):            c = 'lat_lon_{}_{}'.format(int(lat_),int(lon_))              v = []            for y_,y in enumerate(years):                '''                    数据的原因，我们                '''                if y >= 4645 - last_thre_years:                    for m_,m in  enumerate(months):                         v.append(vals[y_,m_,j,i])            df[c] = v    return df
```

```
%%timedf_CMIP_sst  = trans_thre_df(df = df_CMIP_sst,  vals   = np.array(nc_CMIP['sst'][:]),  lats = lats, lons = lons, years = years, months = months)df_CMIP_sst.to_csv(CMIP_trans_path + 'df_CMIP_sst.csv',index = None)del df_CMIP_sstgc.collect()df_CMIP_t300 = trans_thre_df(df = df_CMIP_t300, vals   = np.array(nc_CMIP['t300'][:]), lats = lats, lons = lons, years = years, months = months)df_CMIP_t300.to_csv(CMIP_trans_path + 'df_CMIP_t300.csv',index = None)del df_CMIP_t300gc.collect()df_CMIP_ua   = trans_thre_df(df = df_CMIP_ua,   vals   = np.array(nc_CMIP['ua'][:]),   lats = lats, lons = lons, years = years, months = months)df_CMIP_ua.to_csv(CMIP_trans_path + 'df_CMIP_ua.csv',index = None)del df_CMIP_uagc.collect()df_CMIP_va   = trans_thre_df(df = df_CMIP_va,   vals   = np.array(nc_CMIP['va'][:]),   lats = lats, lons = lons, years = years, months = months)df_CMIP_va.to_csv(CMIP_trans_path + 'df_CMIP_va.csv',index = None)del df_CMIP_vagc.collect()
```

```
(36036, 1729)
```

数据建模

![](https://mmbiz.qpic.cn/mmbiz_png/US10Gcd0tQEfcffueY0reDaT8agHibMbkl6VPJicIaSLBOMT46hKst5wjTztibed2dJsrke6B0nRpRPvJXnC2mlSg/640?wx_fmt=png)

  

![](https://mmbiz.qpic.cn/mmbiz_svg/LwcbhAmMnZBibnjHfa5Dbkk7tP04JCicCSQTxgyGQTicyGIU7M0P5BVzNibgJOO9BU7K91hTpQDPkc1eFEZiaXlGhuoBYTGTn3vc6/640?wx_fmt=svg)

01

工具包导入\&数据读取

1\. 工具包导入

```
import pandas as pdimport numpy  as npimport tensorflow as tffrom tensorflow.keras.optimizers import Adamimport matplotlib.pyplot as pltimport scipy import joblibfrom netCDF4 import Datasetimport netCDF4 as nc from tensorflow.keras.callbacks import LearningRateScheduler, Callbackimport tensorflow.keras.backend as Kfrom tensorflow.keras.layers import *from tensorflow.keras.models import *from tensorflow.keras.optimizers import *from tensorflow.keras.callbacks import *from tensorflow.keras.layers import Input import gc%matplotlib inline     
```

### 2\. 数据读取

#### SODA\_label处理

 1.     标签

```
标签数据为Nino3.4 SST异常指数，数据维度为（year,month）。  CMIP(SODA)_train.nc对应的标签数据当前时刻Nino3.4 SST异常指数的三个月滑动平均值，因此数据维度与维度介绍同训练数据一致注：三个月滑动平均值为当前月与未来两个月的平均值。
```

```
label_path       = './data/SODA_label.nc' nc_label         = Dataset(label_path,'r')tr_nc_labels     = nc_label['nino'][:] 
```

### 2\. 原始特征数据读取

```
SODA_path        = './data/SODA_train.nc'nc_SODA          = Dataset(SODA_path,'r') nc_sst           = np.array(nc_SODA['sst'][:])nc_t300          = np.array(nc_SODA['t300'][:])nc_ua            = np.array(nc_SODA['ua'][:])nc_va            = np.array(nc_SODA['va'][:])
```

![](https://mmbiz.qpic.cn/mmbiz_svg/LwcbhAmMnZBibnjHfa5Dbkk7tP04JCicCSQTxgyGQTicyGIU7M0P5BVzNibgJOO9BU7K91hTpQDPkc1eFEZiaXlGhuoBYTGTn3vc6/640?wx_fmt=svg)

02

模型构建

1\. 神经网络框架  

```
def RMSE(y_true, y_pred):    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))def RMSE_fn(y_true, y_pred):    return np.sqrt(np.mean(np.power(np.array(y_true, float).reshape(-1, 1) - np.array(y_pred, float).reshape(-1, 1), 2)))def build_model():      inp    = Input(shape=(12,24,72,4))          x_4    = Dense(1, activation='relu')(inp)       x_3    = Dense(1, activation='relu')(tf.reshape(x_4,[-1,12,24,72]))    x_2    = Dense(1, activation='relu')(tf.reshape(x_3,[-1,12,24]))    x_1    = Dense(1, activation='relu')(tf.reshape(x_2,[-1,12]))         x = Dense(64, activation='relu')(x_1)      x = Dropout(0.25)(x)     x = Dense(32, activation='relu')(x)       x = Dropout(0.25)(x)      output = Dense(24, activation='linear')(x)       model  = Model(inputs=inp, outputs=output)    adam = tf.optimizers.Adam(lr=1e-3,beta_1=0.99,beta_2 = 0.99)     model.compile(optimizer=adam, loss=RMSE)    return model
```

#### 2\. 训练集验证集划分

```
### 训练特征，保证和训练集一致tr_features = np.concatenate([nc_sst[:,:12,:,:].reshape(-1,12,24,72,1),nc_t300[:,:12,:,:].reshape(-1,12,24,72,1),\                              nc_ua[:,:12,:,:].reshape(-1,12,24,72,1),nc_va[:,:12,:,:].reshape(-1,12,24,72,1)],axis=-1)### 训练标签，取后24个tr_labels = tr_nc_labels[:,12:] ### 训练集验证集划分tr_len     = int(tr_features.shape[0] * 0.8)tr_fea     = tr_features[:tr_len,:].copy()tr_label   = tr_labels[:tr_len,:].copy() val_fea     = tr_features[tr_len:,:].copy()val_label   = tr_labels[tr_len:,:].copy() 
```

#### 3\. 模型训练

```
#### 构建模型model_mlp     = build_model()#### 模型存储的位置model_weights = './model_baseline/model_mlp_baseline.h5'checkpoint = ModelCheckpoint(model_weights, monitor='val_loss', verbose=0, save_best_only=True, mode='min',                             save_weights_only=True)plateau        = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_delta=1e-4, mode='min')early_stopping = EarlyStopping(monitor="val_loss", patience=20)history        = model_mlp.fit(tr_fea, tr_label,                    validation_data=(val_fea, val_label),                    batch_size=4096, epochs=200,                    callbacks=[plateau, checkpoint, early_stopping],                    verbose=2)
```

#### 4\. 模型预测

```
prediction = model_mlp.predict(val_fea)
```

#### 5\. Metrics

```
from   sklearn.metrics import mean_squared_errordef rmse(y_true, y_preds):    return np.sqrt(mean_squared_error(y_pred = y_preds, y_true = y_true))def score(y_true, y_preds):    accskill_score = 0    rmse_scores    = 0    a = [1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6    y_true_mean = np.mean(y_true,axis=0)     y_pred_mean = np.mean(y_preds,axis=0) #     print(y_true_mean.shape, y_pred_mean.shape)    for i in range(24):         fenzi = np.sum((y_true[:,i] -  y_true_mean[i]) *(y_preds[:,i] -  y_pred_mean[i]) )         fenmu = np.sqrt(np.sum((y_true[:,i] -  y_true_mean[i])**2) * np.sum((y_preds[:,i] -  y_pred_mean[i])**2) )         cor_i = fenzi / fenmu            accskill_score += a[i] * np.log(i+1) * cor_i        rmse_score   = rmse(y_true[:,i], y_preds[:,i])#         print(cor_i,  2 / 3.0 * a[i] * np.log(i+1) * cor_i - rmse_score)        rmse_scores += rmse_score         return  2 / 3.0 * accskill_score - rmse_scores 
```

```
print('score', score(y_true = val_label, y_preds = prediction))
```

模型预测

![](https://mmbiz.qpic.cn/mmbiz_png/US10Gcd0tQEfcffueY0reDaT8agHibMbkl6VPJicIaSLBOMT46hKst5wjTztibed2dJsrke6B0nRpRPvJXnC2mlSg/640?wx_fmt=png)

在上面的部分，我们已经训练好了模型，接下来就是提交模型并在线上进行预测，这块可以分为三步：

- 导入模型；
- 读取测试数据并且进行预测；
- 生成提交所需的版本；

  

![](https://mmbiz.qpic.cn/mmbiz_svg/LwcbhAmMnZBibnjHfa5Dbkk7tP04JCicCSQTxgyGQTicyGIU7M0P5BVzNibgJOO9BU7K91hTpQDPkc1eFEZiaXlGhuoBYTGTn3vc6/640?wx_fmt=svg)

01

模型导入

```
import tensorflow as tfimport tensorflow.keras.backend as Kfrom tensorflow.keras.layers import *from tensorflow.keras.models import *from tensorflow.keras.optimizers import *from tensorflow.keras.callbacks import *from tensorflow.keras.layers import Input import numpy as npimport osimport zipfiledef RMSE(y_true, y_pred):    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))def build_model():      inp    = Input(shape=(12,24,72,4))          x_4    = Dense(1, activation='relu')(inp)       x_3    = Dense(1, activation='relu')(tf.reshape(x_4,[-1,12,24,72]))    x_2    = Dense(1, activation='relu')(tf.reshape(x_3,[-1,12,24]))    x_1    = Dense(1, activation='relu')(tf.reshape(x_2,[-1,12]))         x = Dense(64, activation='relu')(x_1)      x = Dropout(0.25)(x)     x = Dense(32, activation='relu')(x)       x = Dropout(0.25)(x)      output = Dense(24, activation='linear')(x)       model  = Model(inputs=inp, outputs=output)    adam = tf.optimizers.Adam(lr=1e-3,beta_1=0.99,beta_2 = 0.99)     model.compile(optimizer=adam, loss=RMSE)    return model model = build_model()model.load_weights('./user_data/model_data/model_mlp_baseline.h5')
```

![](https://mmbiz.qpic.cn/mmbiz_svg/LwcbhAmMnZBibnjHfa5Dbkk7tP04JCicCSQTxgyGQTicyGIU7M0P5BVzNibgJOO9BU7K91hTpQDPkc1eFEZiaXlGhuoBYTGTn3vc6/640?wx_fmt=svg)

02

模型预测

```
test_path = './tcdata/enso_round1_test_20210201/'### 1. 测试数据读取files = os.listdir(test_path)test_feas_dict = {}for file in files:    test_feas_dict[file] = np.load(test_path + file)    ### 2. 结果预测test_predicts_dict = {}for file_name,val in test_feas_dict.items():    test_predicts_dict[file_name] = model.predict(val).reshape(-1,)#     test_predicts_dict[file_name] = model.predict(val.reshape([-1,12])[0,:])### 3.存储预测结果for file_name,val in test_predicts_dict.items():     np.save('./result/' + file_name,val) 
```

  

![](https://mmbiz.qpic.cn/mmbiz_svg/LwcbhAmMnZBibnjHfa5Dbkk7tP04JCicCSQTxgyGQTicyGIU7M0P5BVzNibgJOO9BU7K91hTpQDPkc1eFEZiaXlGhuoBYTGTn3vc6/640?wx_fmt=svg)

03

预测结果打包

```
#打包目录为zip文件（未压缩）def make_zip(source_dir='./result/', output_filename = 'result.zip'):    zipf = zipfile.ZipFile(output_filename, 'w')    pre_len = len(os.path.dirname(source_dir))    source_dirs = os.walk(source_dir)    print(source_dirs)    for parent, dirnames, filenames in source_dirs:        print(parent, dirnames)        for filename in filenames:            if '.npy' not in filename:                continue            pathfile = os.path.join(parent, filename)            arcname = pathfile[pre_len:].strip(os.path.sep)   #相对路径            zipf.write(pathfile, arcname)    zipf.close()make_zip() 
```

提升方向

![](https://mmbiz.qpic.cn/mmbiz_png/US10Gcd0tQEfcffueY0reDaT8agHibMbkl6VPJicIaSLBOMT46hKst5wjTztibed2dJsrke6B0nRpRPvJXnC2mlSg/640?wx_fmt=png)

  

- 模型角度：我们只使用了简单的MLP模型进行建模，可以考虑使用其它的更加fancy的模型进行尝试；
- 数据层面：构建一些特征或者对数据进行一些数据变换等；
- 针对损失函数设计各种trick的提升技巧；