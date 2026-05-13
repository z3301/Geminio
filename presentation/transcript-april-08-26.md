Hey Ze. Hello. Hey Dan. How are you? Good. How are you?
1:301 minute, 30 secondsI'm doing good. Good. Let's wait for Dr. I guess.
1:381 minute, 38 secondsSay again. I guess let's wait for Dr. Jo. Okay.
1:591 minute, 59 secondsSo, how's the project and uh everything?
2:022 minutes, 2 secondsSo, I guess uh the integration was quite simple,
2:072 minutes, 7 secondsright? implementing sync clip and other fusion encoders, robust vision encoders. Uh I'm sorry.
2:152 minutes, 15 secondsUh is my voice coming to uh low?
2:212 minutes, 21 secondsIt's it's a little low on my speaker. I see if I can turn it up a little louder.
2:282 minutes, 28 secondsOkay, that should be a little better.
2:362 minutes, 36 secondsOkay. Now, yeah.
2:382 minutes, 38 secondsOkay. So, uh did you face any difficulties like when integrating the v the robust vision encoders or it was quite small you know?
2:482 minutes, 48 secondsYeah. Okay.
3:083 minutes, 8 secondsHello. How are you? Hi there. Good afternoon. Good afternoon. I'm awesome.
3:153 minutes, 15 secondsHi, Zarif. So, how is everything going then? Uh, really good, I think. Um,
3:243 minutes, 24 secondsso, uh, where do you want to start?
3:283 minutes, 28 secondsYeah, I saw your uh last email where you uh provided the result. I think uh
3:363 minutes, 36 secondsI think uh that's the result that you got from steam clip right then and yesing
3:433 minutes, 43 secondsuh differential privacy base LGD uh fed average multi-pop defense complete defense okay yeah you uh you
3:523 minutes, 52 secondshave I think a significant result uh can you uh first uh discuss you know uh about the result that you got after
3:593 minutes, 59 secondsapplying the sip okay so Um, SIMC clip uh
4:084 minutes, 8 secondsSIM clip doesn't seem to work really uh kind of at all on gradient inversion attacks.
4:174 minutes, 17 secondsOkay. Um uh none of the uh the robust models.
4:244 minutes, 24 secondsum they um you know they they defend against uh adversarial inputs um but
4:324 minutes, 32 secondsthat's not the the threat that uh that gradient inversion um uh attacks, right?
4:414 minutes, 41 secondsUm the gradient inversion attacks at the gradient level. Um and so um what the
4:494 minutes, 49 secondsthe uh robust um what the robust models do is actually
4:554 minutes, 55 secondsmake it um in a lot of cases uh the um the encoder the the vision encoders uh
5:035 minutes, 3 secondslearn more semantically meaningful features and that inadvertently gives the attacker better uh discrimination.
5:135 minutes, 13 secondsUm, and they're allowed to attack.
5:155 minutes, 15 secondsThey're they're able to attack uh uh stronger. Um,
5:225 minutes, 22 secondsso what I did was um I I investigated all of the uh defense uh mechanisms that
5:325 minutes, 32 secondsare um that were uh presented in um the
5:385 minutes, 38 secondsJimno paper. and uh they they tested uh differential privacy but um and they
5:485 minutes, 48 secondsclaimed that it didn't work. Um but what they didn't do is do a calibrated
5:555 minutes, 55 secondsuh DPSGD with epsilon accounting.
6:006 minutesUm, and that's the first thing that I found, the only thing that I found uh that actually stops uh the gradient
6:096 minutes, 9 secondsinversion attacks. And it it does so very well at the um perhaps at the expense of um accuracy.
6:226 minutes, 22 secondsuh depending on on what you set the epsilon value to uh you lose a little
6:296 minutes, 29 secondsbit of accuracy. Um but uh with an epsilon of five uh on the medical domain um you really only lose about one to 2%
6:416 minutes, 41 secondsaccuracy and uh uh you can the the um
6:496 minutes, 49 secondsthe it almost stops the the uh the attack completely on the UAV
6:586 minutes, 58 secondsuh data set uh in epsilon of five completely uh stops the the attack.
7:077 minutes, 7 secondsOkay. Then yeah so you are saying for UAV data set like uh this kind of defense mechanism working very well
7:147 minutes, 14 secondsright? Yes. Very well. Because and it's because um because the uh the the
7:217 minutes, 21 secondsfeatures are so uh uh the the gradients are so large that the the features are
7:287 minutes, 28 secondsso distinct and on um on the chest X-rays the features are much less distinct. you know, it really it takes
7:357 minutes, 35 secondslike a real expert, a radiologist to to to properly read an X-ray to see the the
7:427 minutes, 42 secondsthe the real details. But on uh on UAV imagery, it's pretty easy to see a road
7:507 minutes, 50 secondsand uh a solar panel and a swimming pool and and such.
7:567 minutes, 56 secondsYeah, I think uh then it is great because as long as we are you know uh getting very good uh defense you know uh
8:048 minutes, 4 secondsfor us I think that should be fine and you also mentioned that for medical data set maybe the accuracy could be dropped 1 to 2% right so maybe uh so 1 to 2%
8:178 minutes, 17 secondscompared to the actual accuracy or yes okay so yeah that is also um not very
8:258 minutes, 25 secondsbad like that is also reasonable I think uh uh as as you mentioned like we don't have uh too many variants you know you
8:338 minutes, 33 secondsknow in healthcare data set so yeah I think uh so now uh you mentioned that maybe uh next you would like to apply
8:428 minutes, 42 secondsFML right federated metal learning so yeah maybe Zarif can discuss more about this but I'm not quite sure like uh
8:518 minutes, 51 secondswhether we can apply FML for you know VLM scenario because FML is trained with
8:588 minutes, 58 secondsuh different you know theoretical concept only limited data you know only limited
9:049 minutes, 4 secondsdata and there is metal learning or meta meta training that we need to perform only based on some limited data uh so
9:139 minutes, 13 secondsbut the VLM is already a pre-trained model right so it is already trained large number of data so yeah so Zarif uh would you like to share anything?
9:259 minutes, 25 secondsUh sure profer. So I I guess uh based on on the findings so basically uh that
9:329 minutes, 32 secondsseeing uh the creating inversion kind of improves with the robust encoders. So
9:399 minutes, 39 secondsbasically I guess like now we can give like a pretty good reasoning behind it like why gradient inversion improves on
9:489 minutes, 48 secondsvision encoders because as we've seen with our recent projects result uh with grad uh with uh adversal training the
9:569 minutes, 56 secondsfeatures become more invariant means the features are more stable. uh when you get the features from the vision encoder
10:0410 minutes, 4 secondsor vision encoder when they're more stable you get more stable gradient and with more stable gradient you can extract more better inversion uh images.
10:1510 minutes, 15 secondsSo basically any adversial uh training trained model won't work in in this case
10:2210 minutes, 22 secondsif you want to defend against uh graded inversion attack actually. So uh then you just mentioned you implemented one
10:3010 minutes, 30 secondskind of like a version uh of like a defense mechanism uh where you are seeing some kind of uh like
10:3810 minutes, 38 secondsloss in accuracy in accuracy but it can defend against uh inversion attack. Is that right? Yes.
10:4610 minutes, 46 secondsSo uh did you implement that from another paper or did Jimmy paper have that defense already? So, so um the
10:5610 minutes, 56 secondsJinio paper uh introduced it. It said that um
11:0211 minutes, 2 secondsuh they tried differential privacy and uh they claimed that it failed. Um but
11:1011 minutes, 10 secondstheir experiments didn't use any calibrated uh uh ablations or or any
11:1811 minutes, 18 secondslike uh they they they didn't even go into it. They just said it doesn't work.
11:2311 minutes, 23 secondsAnd what I found is that uh it actually does work and it works very well.
11:2811 minutes, 28 secondsOkay. So uh so the calibrated version of the inversion adder is that uh you took
11:3511 minutes, 35 secondsfrom some paper or uh just you did it by yourself.
11:3911 minutes, 39 secondsUm yes. So uh a body and others in 2016 uh talk about uh differential privacy
11:4611 minutes, 46 secondswith uh stochastic gradient descent. Um it's actually the standard gradient level privacy defense. Um and uh uh so I
11:5611 minutes, 56 secondsimplemented it from there. Um what really uh what really um works out of it
12:0512 minutes, 5 secondsis um is the the the gradient clipping um clipping the L2 normalization um uh
12:1412 minutes, 14 secondsclipping it. So that's why um it actually slightly reduces the accuracy
12:2012 minutes, 20 secondsof the model is because um you you don't
12:2412 minutes, 24 secondsget uh the um you you reduce the uh the
12:3212 minutes, 32 secondsum you reduce the the you clip the the gradients. Um, so, uh,
12:4212 minutes, 42 secondsthe the gradient inversion doesn't work because there isn't the this this large gradient for it to to reconstruct an
12:5112 minutes, 51 secondsimage from. So, that's why it works, but that's also why um, the model learns a little bit less efficiently.
12:5912 minutes, 59 secondsUm, it can still learn. The model the model still works very well. It just may take um uh you know more epochs to train or yeah more data.
13:1013 minutes, 10 secondsOkay.
13:1213 minutes, 12 secondsOkay. Uh also Dan, have you have you have you found uh any other papers or
13:2013 minutes, 20 secondsany other technique that could uh make the attack more defendable than uh
13:2713 minutes, 27 secondsgradient like than gradient clipping or others? Like any other technique you have seen that you want to implement next or you want to improve on?
13:3613 minutes, 36 secondsUm, so no, I mean I like I tried every other uh I tried every other defense
13:4213 minutes, 42 secondsmechanism that the the JNO paper um uh mentions and like uh
13:5213 minutes, 52 secondsuh the the Fed average works uh sometimes and doesn't work other
13:5813 minutes, 58 secondstimes. Uh it's very inconsistent. Um especially on medical data, it's it's to it's completely non-monotonic. It's up and down and all over the place.
14:0814 minutes, 8 secondsUm the uh uh
14:1514 minutes, 15 secondsexcuse me um the uh the
14:2014 minutes, 20 secondsrobust uh clip methods. Um there was one
14:2714 minutes, 27 secondsone which one was it? Um, so fair two uh actually um did No, no, no. I'm sorry.
14:3714 minutes, 37 secondsUh uh um the uh I don't know how you how you say this in conversation, but Takcoa. Okay.
14:4814 minutes, 48 secondsOkay. So, uh so TCOA 2 um actually helped a little bit, but really not even
14:5414 minutes, 54 secondsenough to to bother investigating it. Um uh it it did better because like in terms
15:0315 minutes, 3 secondsof like robustness and it semantic performance takea performs the poorest.
15:0815 minutes, 8 secondsSo okay more code you have it should perform uh much
15:1615 minutes, 16 secondsless in terms of difference and uh that makes sense inversion because then you have more stable features more semantically
15:2515 minutes, 25 secondsmeaningful features and then you you have more stable features from more stable features you have more stable variance you have less in less variance
15:3415 minutes, 34 secondsand then you have uh like pretty good uh picture. So uh that makes sense.
15:4115 minutes, 41 secondsYeah. So so you can so I so we have a paper it's called slate that got published in CVPR. If you see the appendex in the appendex of the paper,
15:5015 minutes, 50 secondsyou will see that we did uh we did something like that like what we did like we took the gradients of the robust
15:5615 minutes, 56 secondsencoder. show that the the gradients of the robos in are more stable and from those robots in you can actually
16:0516 minutes, 5 secondsactually do image to uh text to image generation rather than uh so basically the clip is pretty good at like image to
16:1316 minutes, 13 secondsimage matching or image to text matching but it's not very good at like text to image creation uh like diffusion models
16:2016 minutes, 20 secondsbut yeah so that explains like why doesn't work here uh so I guess uh what
16:2816 minutes, 28 secondswe can do is uh as of now since you have a good starting point with the uh with
16:3516 minutes, 35 secondsdifference mechanism I guess you call it calibrated uh gradient inversion can can repeat
16:4116 minutes, 41 secondsthat again what was it calibrated hello hello
16:4816 minutes, 48 secondsyes yes so uh so uh you implemented the calibrated version of uh differential privacy calibrated differential privacy, right?
16:5816 minutes, 58 secondsYeah. Yep.
17:0017 minutesOkay. So, right now you're seeing that you can totally defend against grad inversion like totally defend against it with some with some uh accuracy drop,
17:1317 minutes, 13 secondsright? Yes. Okay.
17:1817 minutes, 18 secondsAnd can you Yes. I've actually got um I've got uh really like quite an extensive ablation study that uh that
17:2717 minutes, 27 secondstests um uh epsilon at uh everything from zero all the way up to I think 100.
17:3617 minutes, 36 secondsOkay. And at which epsilon it works works the best?
17:4117 minutes, 41 secondsUm the the best tradeoff really for for both uh is really about an epsilon of five.
17:5017 minutes, 50 secondsFive. Okay. Yeah.
17:5517 minutes, 55 secondsSo are we using this like uh boundary like epsilon boundary only for
18:0318 minutes, 3 secondsuh differential privacy or you're using similar kind of boundary also in German.
18:1018 minutes, 10 secondsI guess German should use some kind of boundary right.
18:1618 minutes, 16 secondsUh yes and I uh I don't know that off the top of my head.
18:2118 minutes, 21 secondsUh Dan, can you can you just uh show me uh the Jino parameters you're using? Uh
18:2818 minutes, 28 secondsthat should be uh that should be okay like so when you're running JNO at what are the parameters you're giving to JO
18:3718 minutes, 37 secondsso if I can see that then I can uh understand.
18:4418 minutes, 44 secondsOkay.
19:3819 minutes, 38 secondsSo I think yeah that has got some interesting findings. For example, now uh we understand that the robust veance
19:4819 minutes, 48 secondsuh you know uh in this kind of attack scenario like graded inversion attack uh they are making this attack more stronger right?
19:5619 minutes, 56 secondsYes.
19:5719 minutes, 57 secondsYeah. So that is a good finding. uh and interesting thing is that like for healthcare domain I can see that uh data
20:0620 minutes, 6 secondsis getting stronger even it is getting maximum uh
20:1320 minutes, 13 secondsyeah maximum I think attack strength in sync clip right sync clip 2 and if you
20:2020 minutes, 20 secondslook at the UAV result uh I think it is better in sync clip
20:2720 minutes, 27 secondsbut uh for fair I think yeah it is the OS uh so
20:3520 minutes, 35 secondsyeah so overall I think uh the current and robust VLM are making the attack
20:4120 minutes, 41 secondsmore stronger um like particularly the gradient inversion attack so this is an interesting finding that we can use
20:5020 minutes, 50 secondsmaybe we can when we think about a paper we can say that maybe the robot VLM backfired something like that so that could make the idea more interesting.
21:0221 minutes, 2 secondsYes, professor. And for I guess uh is a calibrated version working pretty good then I guess like uh I'm not sure. Uh so
21:1121 minutes, 11 secondsI guess the calibrated version is uh what Dan implemented. So this is kind of
21:1921 minutes, 19 secondsif if we you know uh do proper analysis uh regarding the clip thing and uh I think as we are getting very good you
21:2721 minutes, 27 secondsknow attack strength I think that should be sufficient enough like if you if we uh plot you know the paper like uh maybe
21:3621 minutes, 36 secondsa clipping is enough or something like that and say that we can mitigate the vision language gradient inversion via
21:4421 minutes, 44 secondsuh this kind of nor nonbounded updates or something like that. We can Yes. Yes.
21:5221 minutes, 52 secondsYeah. I think uh that clipping is enough for green version that could be the title and then you can
21:5921 minutes, 59 secondsand then yeah and we can also you know uh maybe uh think about a different name for DPSG
22:0922 minutes, 9 secondsmaybe uh that can also make the things more interesting. Maybe we can think about it for VLM. Maybe we can rename
22:1822 minutes, 18 secondsit. Yeah. Something else that can make also things more interesting.
22:2722 minutes, 27 secondsOkay, I'm going to share my screen. Yeah, sure.
22:5122 minutes, 51 secondsMaybe. There we go.
22:5522 minutes, 55 secondsOkay. Um, probably really small. I will adjust it as necessary.
23:0223 minutes, 2 secondsYeah, I I we can see it.
23:0823 minutes, 8 secondsOh, I just removed it. Um, how do I hold on?
23:2623 minutes, 26 secondsOkay, that readable.
23:3323 minutes, 33 secondsYes. Can you see it?
23:4023 minutes, 40 secondsYes, it's all right. I can see it.
23:4423 minutes, 44 secondsQuestion levels.
23:5823 minutes, 58 secondsNo, she's not ready to go.
24:0324 minutes, 3 secondsSo, uh Dan, what are you passing for the query?
24:0824 minutes, 8 secondsUm so the queries are
24:3924 minutes, 39 secondsUm also then like when you're using like high epsylon what do you see? uh do you
24:4624 minutes, 46 secondssee like more defense or do you see so if you are using like high epsylon it's
24:5224 minutes, 52 secondsquite natural that you lose accuracy so with high epsylon I guess accuracy is dipping but what what do you what do you
25:0025 minutessee in the in the defense mechanism is it uh improving or you max out at five
25:0725 minutes, 7 secondsokay that's okay that should answer Also
25:1925 minutes, 19 secondsDan uh one other question. So the calibrated part uh did you do it by yourself? I mean this is is it your introduction
25:2625 minutes, 26 secondslike uh did you uh uh like you know code this how how do I say like is the
25:3425 minutes, 34 secondscalibration part the novely of this project or do you pick it from other paper or like someone already has
25:4125 minutes, 41 secondspublished the calibration part uh no that's that's novel that's novel right okay so so that that
25:4925 minutes, 49 secondsso so so professor in that case I guess that's it I guess we have the finding, we have the defense and we have the code I guess.
26:0026 minutesYes, professor.
26:0126 minutes, 1 secondI think then uh what we could do like uh I think uh this idea is good enough you know uh we can submit it maybe to EMNLP.
26:0926 minutes, 9 secondsSo maybe I I can share share one of one conference link and another thing is like a like we can as I mentioned we can
26:1826 minutes, 18 secondsrename you know the DPSG to something like maybe uh I don't know like uh something
26:2826 minutes, 28 secondsdifferent maybe norm bounded federated update something like that or NBFU something like that where we uh maybe the code thing would be the clipping you
26:3726 minutes, 37 secondsknow and yeah and then it would more align with the federated update maybe so
26:4626 minutes, 46 secondsokay cool so also uh Dan uh Dan has already has made the calibration part So basically
26:5426 minutes, 54 secondscalibrated and then we can have a different name. Yeah.
27:0027 minutesAnd yeah I think also also we can think about like uh maybe we need yeah we can do more analysis regarding the gradient
27:0927 minutes, 9 secondsinversion success with the variance you know or the gap.
27:1327 minutes, 13 secondsYeah. So yeah maybe a simple an analytical argument you know we can also provide in the paper. So yeah, we can we can do that I think side by side.
27:2627 minutes, 26 secondsUh uh also Dan uh I guess did you just use Jio uh as an attack or did you use uh some other attacks also?
27:3927 minutes, 39 secondsNo, just a minute. Just a minute.
27:4227 minutes, 42 secondsI think that's the important part like Zarif. I think we need to maybe if we can also consider another three or four.
27:5227 minutes, 52 secondsYeah, I think that should be good enough.
27:5627 minutes, 56 secondsSo, so Dan then I guess like what what we can do is uh you can take a look into the Ju paper and I guess like you can
28:0328 minutes, 3 secondsfind a few more uh gradient inversion attack based on clip and
28:1028 minutes, 10 secondstry to run it on the same codebase like same data that professor shared I guess Vante and uh Mlip and then use your
28:1828 minutes, 18 secondsdefense mechanism which is the calibrated DPSJD and then see like If
28:2528 minutes, 25 secondsour defense your defense mechanism works with those attacks also. So if we can just showcase like our defense mechanism
28:3328 minutes, 33 secondsworks with three or four gradient inversion attack then I guess we're good to go. Okay.
28:4128 minutes, 41 secondsI think and then uh the link that I shared that is one of the top conferences you know uh in computer science
28:4828 minutes, 48 secondsI think uh the deadline is May 27. So as we already got a solution you know it is working very good you know uh in
28:5728 minutes, 57 secondsdifferent robust VLMs and we just tested with u one particular gradient inversion attack. So uh maybe we can we can stick
29:0729 minutes, 7 secondswith the same robot VM that you are currently dealing with maybe simply fair ta that should be good enough. We just have to you know uh test it uh with
29:1629 minutes, 16 secondsother different gradient inversion track and then uh we should be good enough you know to submit it as a paper.
29:2329 minutes, 23 secondsOkay. Um so these are the queries.
29:2929 minutes, 29 secondsThese are the medical queries. These are the UAV queries. Okay.
29:3629 minutes, 36 secondsOkay. Okay.
29:4029 minutes, 40 secondsSo, so these are the questions kind of like uh your question. Yes. Okay. Okay. Sounds good. Okay.
29:4929 minutes, 49 secondsYes.
29:5029 minutes, 50 secondsYeah. Zarif like right now I think uh also we can think about maybe uh I don't know exactly which particular UAV data
29:5729 minutes, 57 secondsset Dan currently used. Maybe uh we can also use the other data set that you shared. Maybe that can also
30:0630 minutes, 6 secondsuh be Yeah, Zarif, you seem to be surprised that I only had one UAV data set.
30:1430 minutes, 14 secondsYes. So, so I like you working with multiple UV data set. Uh so then what we
30:2130 minutes, 21 secondscan do is like uh for our immediate uh next approach or action uh let's go with
30:2930 minutes, 29 secondsuh two or three attics. Okay. So since we have like two data set I guess like
30:3730 minutes, 37 secondswe're in a good stage as of now but what we need uh immediately is like uh we need to work on uh two or three
30:4530 minutes, 45 secondsinversion attack based on flip. So uh uh I guess like from Germany you can find those or also I can take a look into it.
30:5330 minutes, 53 secondsIf I find some interesting inverion attack based on clip and f learn then I can sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh
30:5730 minutes, 57 secondssh sh sh sh sh sh sh sh sh sh sh sh sh sh sh share them also.
30:5830 minutes, 58 secondsOkay. But I guess like I guess you have some uh prevented imperson attacks already like other than Germany I guess like if you see the related to works in
31:0631 minutes, 6 secondsGermany or they have already mentioned few other uh very imperson attack right so uh you can start with any one of them
31:1431 minutes, 14 secondsand implement and see how those work with your defense mechanism. Okay. Okay.
31:2231 minutes, 22 secondsOkay. And then we can introduce new data set uh afterwards. Maybe another one like one other or maybe two other data should be fine.
31:3331 minutes, 33 secondsOkay.
31:3531 minutes, 35 secondsYeah. And then like uh try to you know save all the result that uh you are getting for example during the evolution
31:4331 minutes, 43 secondsor implementation because we need to provide all the result as a table you know uh during paper submission. So,
31:5431 minutes, 54 secondsso uh professor uh what I can do is like I can just create a GitHub repo in our lab
32:0132 minutes, 1 secondlab uh handle and I can add Dan as a collaborator and myself as a collaborator and then whenever you have
32:1032 minutes, 10 secondssome data uh or results just push them to the repository so that like we can also check meanwhile and so like
32:1732 minutes, 17 secondseverything is log okay because uh I guess some sometimes uh the PGX or maybe uh the other one is
32:2532 minutes, 25 secondsA 100 that may become unavailable because there something happened with me once last December my log data was kind
32:3532 minutes, 35 secondsof uh held by well for like two weeks so I guess that should be better right okay
32:4332 minutes, 43 secondsyeah that's that's that sounds great yeah okay so Dan uh on teams can I just share your GitHub handle so that I can add uh add you in our GitHub background.
32:5332 minutes, 53 secondsSure.
33:0033 minutesOkay. Sounds great. I think and Dan like uh could you please also share this maybe recording with George as well?
33:0633 minutes, 6 secondsYes, absolutely. I I am I am recording this. Um uh it should automatically send you guys a a
33:1533 minutes, 15 secondssummary and um I will uh I will make sure that George has uh access to the recording and the summary and uh see what he says.
33:2533 minutes, 25 secondsYeah. Excellent. Yeah, sounds great.
33:2633 minutes, 26 secondsYeah, you are doing a great job. I think so. Yeah. So, I'm happy to be working with you guys.
33:3433 minutes, 34 secondsYeah, thank you. Yeah, it's our pleasure. Yeah. Thanks, Dan. Thank you, Zarif.
33:4133 minutes, 41 secondsAll right. Thank you. Talk soon.

# Main purpose
- Discuss research findings on Vision Language Model (VLM) gradient inversion attacks and defense mechanisms
- Review experimental results from testing robust vision encoders (SimClip, FAIR, Takoa) against gradient inversion attacks
- Evaluate the effectiveness of calibrated differential privacy as a defense mechanism
- Plan next steps for paper submission to EMNLP conference


# Project status


Issue/Ticket
Status

## Robust Vision Encoder Testing
- Testing completed on SimClip, FAIR, and Takoa encoders
- Finding: Robust models make gradient inversion attacks stronger, not weaker
- Robust encoders learn more semantically meaningful features, inadvertently improving attacker discrimination
- Takoa 2 showed minimal improvement, insufficient for further investigation
## Calibrated Differential Privacy Defense (DPSGD)
- Successfully implemented calibrated DPSGD with epsilon accounting
- Optimal epsilon value identified at 5 for both datasets
- UAV dataset: Attack completely stopped with epsilon of 5
- Medical dataset: 1-2% accuracy drop with epsilon of 5, attack nearly stopped
- Defense works through gradient clipping and L2 normalization
- Novel calibration approach developed by Dan
- Extensive ablation study completed testing epsilon values from 0 to 100
## Dataset Performance Analysis
- UAV dataset shows distinct features (roads, solar panels, swimming pools), making defense highly effective
- Medical chest X-ray dataset has less distinct features requiring expert analysis, resulting in slightly lower defense effectiveness
- Currently working with one UAV dataset and one medical dataset
- Decision: Consider adding 1-2 additional datasets for paper submission
## Alternative Defense Mechanisms
- Fed average tested: Inconsistent results, non-monotonic on medical data
- All defense mechanisms from Geminio paper evaluated
- Only calibrated differential privacy proved effective
## Other topics
- Paper Submission Planning: Target EMNLP conference with May 27 deadline
- Naming Convention: Consider renaming DPSGD to something more distinctive, such as "norm-bounded failure rate update" or similar
- Paper Positioning: Frame findings as "robust VLMs backfire" in gradient inversion attack scenarios
- Analytical Component: Need to add analysis regarding gradient inversion success correlation with variance
- GitHub Repository: Set up lab GitHub repository for result logging and collaboration to prevent data loss

# Action items
- Dan
    [] Implement 2-3 additional gradient inversion attacks based on CLIP from Geminio paper's related works
    [] Test calibrated DPSGD defense mechanism against the new attacks
    [] Save all evaluation and implementation results in table format for paper submission
    [] Share GitHub handle with Zarif on Teams
    [] Push all data and results to the GitHub repository regularly
    [] Share meeting recording and summary with George
- Zarif
    [] Create GitHub repository under lab handle
    [] Add Dan as collaborator to the repository
    [] Research and share additional gradient inversion attacks based on CLIP if found
    [] Review the Slay paper appendix showing gradient stability in robust encoders
- Ahmed
    [] Share EMNLP conference link and details with team
    [] Review and provide feedback on additional gradient inversion attack implementations
- Team
    [] Develop simple analytical argument regarding gradient inversion success and variance for paper
    [] Consider adding 1-2 additional datasets for comprehensive evaluation
    [] Finalize naming convention for the defense mechanism
    [] Prepare paper submission for EMNLP by May 27 deadline