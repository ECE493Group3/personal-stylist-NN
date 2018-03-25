import os

ANNOTATION_FILE = os.path.join('DATA', 'Anno', 'list_attr_img.txt')

def recalculate_weights():
    n_positives = [0] * 1000

    with open(ANNOTATION_FILE) as f:
        lines = f.readlines()
        n_images = int(lines[0])

        for line in lines[2:]:
            cols = line.split()
            for i, att in enumerate(cols[1:]):
                if att == '1':
                    n_positives[i] += 1

        print(n_positives)
        positive_ratios = [float(n)/n_images for n in n_positives]

        total_positive_ratio = float(sum(n_positives)) / (1000 * n_images)

        print("Total positive ratio is {}".format(total_positive_ratio))

        print("Positive ratio for each attribute:")
        print(positive_ratios)

TOTAL_POSITIVE_RATIO = 0.003322741700147292

ATTRIBUTE_WEIGHTS = [0.011202467308849258, 0.019510963896245792, 0.00042873640317818147, 0.0001936228917578884, 0.00018325023684228725, 0.001500577411123635, 0.0007053405342608792, 0.0008298123932480931, 0.00042182129990111405, 0.00021091064995055702, 0.0006258168465746036, 0.007506344607256709, 0.0006119866400204687, 0.00044602416137085006, 0.007347297231884159, 0.005995394541217473, 0.00035612781876897333, 0.0018428750233384736, 0.003972726832675246, 0.002323474701094661, 0.0038033068023870936, 0.0008713030129104978, 0.0005774111236351315, 0.0016976578545200573, 0.00288014051489859, 0.00421821299901114, 0.000670765017875542, 0.0035163300163887946, 0.0007744915670315536, 0.0006776801211526094, 0.00459854367924985, 0.00019016534011935467, 0.0005255478490571256, 0.0002593163728900291, 0.0005428356072497943, 0.0002558588212514954, 0.003917406006458707, 0.0007675764637544862, 0.0037825614925558913, 0.0020503281216504967, 0.0010718410079454536, 0.0009542842522353071, 0.003955439074482577, 0.001863620333169676, 0.006500197080443397, 0.0051068037701143066, 0.00044256660973231637, 0.0033365373311850413, 0.0059608190248321355, 0.0008574728063563629, 0.009342304527318115, 0.0005774111236351315, 0.0002247408565046919, 0.0002247408565046919, 0.00029389188927536635, 0.0007675764637544862, 0.00029734944091390004, 0.0004010759900699117, 0.003305419366438238, 0.0003146371991065687, 0.0013380724841125502, 0.012378034865950722, 0.00024894371797442796, 0.0003215523023836361, 0.00020053799503495586, 0.0003042645441909675, 0.0005186327457800583, 0.00022128330486615819, 0.0016838276479659224, 0.0006915103277067443, 0.004270076273589146, 0.0005393780556112606, 0.002147139567529441, 0.0020779885347587666, 0.0010614683530298524, 0.0001936228917578884, 0.001116789179246392, 0.0014971198594851015, 0.00047714212611765355, 0.0003284674056607035, 0.0015662708922557758, 0.02219402396774796, 0.001241261038233606, 0.007354212335161226, 0.0017771815422063329, 0.0010026899751747792, 0.0003146371991065687, 0.005770653684712781, 0.00020399554667348957, 0.00021436820158909073, 0.0024375739051662737, 0.0011237042825234594, 0.00019708044339642212, 0.01284826188879131, 0.0006431046047672722, 0.0002593163728900291, 0.0001797926852037535, 0.00029389188927536635, 0.0003388400605763047, 0.006918560828705977, 0.0014106810685217584, 0.0005532082621653954, 0.0003146371991065687, 0.001192855315294134, 0.002001922398711025, 0.0012378034865950724, 0.0005601233654424629, 0.0003388400605763047, 0.004221670550649674, 0.0014936623078465677, 0.0023165595978175933, 0.0010752985595839874, 0.01748483863606503, 0.0010856712144995885, 0.0022370359101313176, 0.006106036193650552, 0.00439454813257636, 0.0010096050784518466, 0.0012758365546189432, 0.002122936706059705, 0.00035612781876897333, 0.014352296851553478, 0.0002454861663358942, 0.0006016139851048676, 0.0027383808977187074, 0.0010580108013913188, 0.0012308883833180048, 0.0012378034865950724, 0.0010787561112225212, 0.005597776102786095, 0.00019016534011935467, 0.0029838670640546018, 0.002084903638035834, 0.0076515617760751255, 0.0001936228917578884, 0.0005566658138039291, 0.0007744915670315536, 0.0005981564334663339, 0.00031117964746803494, 0.0050756858053675035, 0.001303496967727213, 0.007257400889282281, 0.04057436847819322, 0.00037687312860017566, 0.00044256660973231637, 0.000273146579444164, 0.002306186942901992, 0.0002454861663358942, 0.0005532082621653954, 0.0005324629523341931, 0.0019189411593862153, 0.0003872457835157768, 0.0006223592949360699, 0.0035543630844126657, 0.0032604711951372996, 0.001317327174281348, 0.000335382508937771, 0.0010165201817289142, 0.0002869767859982989, 0.0032466409885831644, 0.000608529088381935, 0.00021782575322762445, 0.024839050971226254, 0.00042182129990111405, 0.0003146371991065687, 0.0003008069925524338, 0.0009300813907655711, 0.0007260858440920815, 0.0006984254309838117, 0.000273146579444164, 0.0003284674056607035, 0.0004321939548167152, 0.0003146371991065687, 0.0004875147810332547, 0.001438341481630028, 0.0005048025392259233, 0.0011202467308849258, 0.0006431046047672722, 0.0010580108013913188, 0.0006396470531287385, 0.000992317320259178, 0.01887131684311705, 0.00023165595978175935, 0.002510182489575482, 0.005175954802884981, 0.000397618438431378, 0.009093360809343688, 0.0002904343376368326, 0.002745296000995775, 0.0031982352656436926, 0.0001936228917578884, 0.0077207128088458005, 0.0013311573808354829, 0.00019016534011935467, 0.0005013449875873896, 0.000608529088381935, 0.025482155575993527, 0.00025240126961296165, 0.0004114486449855129, 0.0006154441916590024, 0.0005670384687195304, 0.002928546237838062, 0.0023822530789497343, 0.0021298518093367724, 0.011949298462772541, 0.0003665004736845745, 0.0007779491186700873, 0.009269695942908907, 0.006216677846083631, 0.0009266238391270374, 0.0018221297135072713, 0.0002904343376368326, 0.031073016575502554, 0.00034229761221483843, 0.00019708044339642212, 0.00047714212611765355, 0.0003699580253231082, 0.0006258168465746036, 0.004283906480143281, 0.0008021519801398234, 0.0002696890278056303, 0.0002627739245285628, 0.0016492521315805851, 0.001102958972692257, 0.0023649653207570656, 0.00032500985402216984, 0.021509428743318283, 0.012813686372405972, 0.00019016534011935467, 0.00023511351142029306, 0.0005151751941415245, 0.004823284535754541, 0.0010787561112225212, 0.0002558588212514954, 0.0014141386201602922, 0.0011202467308849258, 0.013850951863966088, 0.00881675667826099, 0.001500577411123635, 0.012101430734868026, 0.00020745309831202328, 0.00045293926464791753, 0.0003665004736845745, 0.0007848642219471548, 0.0008367274965251606, 0.005469846692160348, 0.002060700776566098, 0.0006845952244296769, 0.0003907033351543105, 0.006444876254226857, 0.0008920483227417001, 0.0014867472045695003, 0.0001763351335652198, 0.0001797926852037535, 0.04556361549259738, 0.00026623147616709655, 0.0006673074662370082, 0.00023165595978175935, 0.001192855315294134, 0.0004494817130093838, 0.00021436820158909073, 0.0007157131891764803, 0.00300115482224727, 0.0002593163728900291, 0.004761048606260934, 0.001500577411123635, 0.0002696890278056303, 0.00023511351142029306, 0.012436813243805796, 0.0001936228917578884, 0.0003526702671304396, 0.0008125246350554246, 0.00285248010179032, 0.0021575122224450423, 0.0009369964940426386, 0.0012827516578960106, 0.006973881654922516, 0.0021402244642523736, 0.0003907033351543105, 0.002492894731382813, 0.0005739535719965977, 0.00026623147616709655, 0.01317327174281348, 0.0024721494215516108, 0.0018394174716999398, 0.0007675764637544862, 0.000273146579444164, 0.006887442863959173, 0.00023165595978175935, 0.0002454861663358942, 0.00020745309831202328, 0.0016561672348576527, 0.015863246917592715, 0.0018152146102302037, 0.00019708044339642212, 0.0002904343376368326, 0.0013104120710042805, 0.0002627739245285628, 0.00038033068023870934, 0.0008159821866939583, 0.008609303579948966, 0.00020745309831202328, 0.0026484845551168305, 0.0017287758192668608, 0.0002869767859982989, 0.000546293158888328, 0.004937383739826154, 0.00210910649950557, 0.001102958972692257, 0.005335002178257532, 0.02289244939873177, 0.00023857106305882677, 0.0009093360809343688, 0.0009058785292958351, 0.00020399554667348957, 0.0014763745496538991, 0.0002281984081432256, 0.0010372654915601165, 0.0012205157284024036, 0.0003872457835157768, 0.00023857106305882677, 0.006106036193650552, 0.0004702270228405861, 0.0002766041310826977, 0.0029078009280068596, 0.003824052112218296, 0.00025240126961296165, 0.002994239718970203, 0.002423743698612139, 0.0002420286146973605, 0.0036338867720989414, 0.000881675667826099, 0.0006292743982131374, 0.0004702270228405861, 0.0012274308316794712, 0.025043046517899743, 0.005556285483123691, 0.01472571242851512, 0.0003699580253231082, 0.0006776801211526094, 0.00020053799503495586, 0.000359585370407507, 0.00042527885153964773, 0.0001936228917578884, 0.00020053799503495586, 0.00018325023684228725, 0.0009750295620665094, 0.0009715720104279757, 0.00035612781876897333, 0.001030350388283049, 0.0003042645441909675, 0.0005704960203580641, 0.0007537462572003513, 0.02589360422097904, 0.0018117570585916701, 0.00020053799503495586, 0.004498274681732372, 0.00020399554667348957, 0.016738007482141746, 0.0021782575322762446, 0.002520555144491083, 0.0016699974414117874, 0.0005013449875873896, 0.0003146371991065687, 0.0008021519801398234, 0.08363125903285365, 0.0001763351335652198, 0.0002593163728900291, 0.0076895948440989965, 0.00021782575322762445, 0.0002281984081432256, 0.00025240126961296165, 0.0012896667611730782, 0.00032500985402216984, 0.0002558588212514954, 0.0002558588212514954, 0.0003837882318772431, 0.0006361895014902047, 0.00020399554667348957, 0.0003077220958295012, 0.026139090387314934, 0.0005739535719965977, 0.00018670778848082096, 0.0003388400605763047, 0.00044602416137085006, 0.001102958972692257, 0.0004667694712020524, 0.0001763351335652198, 0.0006431046047672722, 0.002181715083914778, 0.00029734944091390004, 0.0009577418038738409, 0.00578794144290545, 0.0020157526052651596, 0.0003146371991065687, 0.000670765017875542, 0.002430658801889206, 0.001739148474182462, 0.0004944298843103221, 0.0006223592949360699, 0.0006603923629599408, 0.000881675667826099, 0.0006880527760682105, 0.00024894371797442796, 0.0002420286146973605, 0.0011824826603785328, 0.0004010759900699117, 0.0016043039602796469, 0.0007848642219471548, 0.0077829487383394074, 0.004612373885803984, 0.0017910117487604678, 0.00024894371797442796, 0.007883217735856885, 0.002842107446874719, 0.007126013927018, 0.0023131020461790597, 0.00039416088679284424, 0.00104072304319865, 0.0013864782070520222, 0.00036304292204604075, 0.0008540152547178293, 0.00018325023684228725, 0.003896660696627504, 0.0004114486449855129, 0.0002766041310826977, 0.00033192495729923727, 0.00019708044339642212, 0.00035612781876897333, 0.006894357967236241, 0.0004010759900699117, 0.002634654348562696, 0.0002904343376368326, 0.0015489831340631073, 0.0011271618341619932, 0.0005255478490571256, 0.0003872457835157768, 0.006804461624634364, 0.008080298179253306, 0.0002558588212514954, 0.0009231662874885036, 0.02397120550995429, 0.0012101430734868025, 0.0001797926852037535, 0.0013795631037749549, 0.0006742225695140757, 0.00228544163307079, 0.0006050715367434012, 0.00023165595978175935, 0.002181715083914778, 0.004204382792457005, 0.0007986944285012896, 0.0013069545193657467, 0.008308496587396533, 0.000757203808838885, 0.00019016534011935467, 0.0002454861663358942, 0.011797166190677057, 0.0005290054006956594, 0.0002420286146973605, 0.0004805996777561873, 0.0014003084136061572, 0.0005739535719965977, 0.0007641189121159525, 0.0001797926852037535, 0.0010960438694151897, 0.00018325023684228725, 0.0011202467308849258, 0.0028213621370435167, 0.0016285068217493828, 0.01368498938531647, 0.0004633119195635187, 0.0004805996777561873, 0.001154822247270263, 0.0023165595978175933, 0.00104072304319865, 0.002071073431481699, 0.00019016534011935467, 0.0013622753455822863, 0.00020399554667348957, 0.0038033068023870936, 0.0006673074662370082, 0.0029008858247297925, 0.0016181341668337816, 0.0002904343376368326, 0.0010338079399215827, 0.00043910905809378263, 0.0009024209776573013, 0.0002766041310826977, 0.002409913492058004, 0.0004356515064552489, 0.00023511351142029306, 0.000795236876862756, 0.0005532082621653954, 0.0006500197080443397, 0.06395779020959678, 0.0001936228917578884, 0.0002904343376368326, 0.0009266238391270374, 0.0003526702671304396, 0.0006915103277067443, 0.0008090670834168908, 0.00029389188927536635, 0.00029389188927536635, 0.00024894371797442796, 0.001255091244787741, 0.00018670778848082096, 0.0004045335417084454, 0.00028006168272123145, 0.00042182129990111405, 0.0012481761415106736, 0.0004356515064552489, 0.005148294389776711, 0.07065161018179807, 0.00019708044339642212, 0.001255091244787741, 0.00034575516385337217, 0.0003734155769616419, 0.0008298123932480931, 0.0015731859955328432, 0.0018117570585916701, 0.0014348839299914945, 0.000781406670308621, 0.00045293926464791753, 0.00019016534011935467, 0.0007606613604774187, 0.001389935758690556, 0.0001797926852037535, 0.0013311573808354829, 0.0008298123932480931, 0.0004114486449855129, 0.004453326510431433, 0.0022508661166854528, 0.000670765017875542, 0.0008367274965251606, 0.0010510956981142512, 0.00047368457447911986, 0.0002281984081432256, 0.0038309672154953635, 0.0005117176425029908, 0.001763351335652198, 0.000670765017875542, 0.0005566658138039291, 0.006396470531287385, 0.002572418419069089, 0.001500577411123635, 0.025357683717006314, 0.0005082600908644571, 0.001030350388283049, 0.00023857106305882677, 0.0005635809170809966, 0.00021436820158909073, 0.00021782575322762445, 0.0005428356072497943, 0.00021436820158909073, 0.00021436820158909073, 0.00023857106305882677, 0.000273146579444164, 0.00034921271549190585, 0.00041490619662404657, 0.006334234601793778, 0.0037134104597852167, 0.004256246067035011, 0.00866808195780404, 0.00018670778848082096, 0.010548990049166384, 0.0013691904488593537, 0.002928546237838062, 0.02301346370608045, 0.0031325417845115516, 0.0008090670834168908, 0.010988099107260167, 0.0003077220958295012, 0.0004563968162864512, 0.009363049837149319, 0.000273146579444164, 0.0011997704185712013, 0.020544771836167375, 0.00044602416137085006, 0.0009162511842114362, 0.00034229761221483843, 0.00019016534011935467, 0.0012516336931492071, 0.0003526702671304396, 0.0002420286146973605, 0.0011133316276078582, 0.0006811376727911432, 0.0014902047562080339, 0.000397618438431378, 0.0007537462572003513, 0.00040799109334697914, 0.000819439738332492, 0.000656934811321407, 0.009418370663365858, 0.0002281984081432256, 0.000273146579444164, 0.05528625070015421, 0.0015386104791475061, 0.0008989634260187676, 0.004325397099805685, 0.00024894371797442796, 0.017664631321268782, 0.0004321939548167152, 0.0015074925144007026, 0.0008021519801398234, 0.00025240126961296165, 0.0015593557889787084, 0.0002558588212514954, 0.007067235549162927, 0.001638879476664984, 0.0011237042825234594, 0.0008505577030792955, 0.001452171688184163, 0.0003734155769616419, 0.0014487141365456292, 0.0005117176425029908, 0.016112190635567143, 0.00023511351142029306, 0.0033711128475703784, 0.002855937653428854, 0.02097005068770702, 0.00047714212611765355, 0.0002593163728900291, 0.0003284674056607035, 0.00394852397120551, 0.0008505577030792955, 0.0004875147810332547, 0.0007744915670315536, 0.0019016534011935468, 0.0014487141365456292, 0.0002454861663358942, 0.001389935758690556, 0.00023857106305882677, 0.0007468311539232838, 0.0010649259046683862, 0.0005220902974185919, 0.0007295433957306152, 0.000273146579444164, 0.005186327457800582, 0.0002247408565046919, 0.01924127486844016, 0.0004702270228405861, 0.0017149456127127259, 0.00036304292204604075, 0.0003734155769616419, 0.0002420286146973605, 0.00032500985402216984, 0.00029734944091390004, 0.004640034298912255, 0.0037064953565081496, 0.0005635809170809966, 0.001317327174281348, 0.0002696890278056303, 0.006268541120661637, 0.002036497915096362, 0.0004702270228405861, 0.00024894371797442796, 0.0012827516578960106, 0.0017598937840136643, 0.0010787561112225212, 0.0009162511842114362, 0.0003146371991065687, 0.0026139090387314935, 0.0004944298843103221, 0.0003907033351543105, 0.0006465621564058059, 0.003727240666339352, 0.002492894731382813, 0.0002593163728900291, 0.0024168285953350714, 0.0005255478490571256, 0.001984634640518356, 0.0001797926852037535, 0.0004010759900699117, 0.0034990422581961263, 0.00023511351142029306, 0.007616986259689789, 0.0008782181161875653, 0.0012343459349565386, 0.0004321939548167152, 0.0008263548416095595, 0.012160209112723098, 0.005089516011921638, 0.0033953157090401147, 0.0008263548416095595, 0.00018670778848082096, 0.0014936623078465677, 0.0018947382979164794, 0.007883217735856885, 0.005542455276569555, 0.0004978874359488559, 0.0035439904294970646, 0.009964663822254185, 0.0002766041310826977, 0.0025862486256232236, 0.011631203712027438, 0.0025897061772617576, 0.007703425050653132, 0.01273762023635823, 0.006016139851048675, 0.0011340769374390608, 0.003063390751740877, 0.0005255478490571256, 0.001403765965244691, 0.0007917793252242223, 0.022712656713528016, 0.0008920483227417001, 0.00028006168272123145, 0.00273146579444164, 0.00022128330486615819, 0.001452171688184163, 0.0007779491186700873, 0.001528237824231905, 0.01211526094142216, 0.0016354219250264504, 0.0007226282924535478, 0.003924321109735774, 0.02335576131829529, 0.0004494817130093838, 0.00047368457447911986, 0.017955065658905617, 0.0008713030129104978, 0.012053025011928554, 0.0033814855024859795, 0.0007883217735856885, 0.0021298518093367724, 0.00039416088679284424, 0.00136573289722082, 0.0006154441916590024, 0.0007641189121159525, 0.12919833207708958, 0.0003665004736845745, 0.00020745309831202328, 0.000484057229394721, 0.0013968508619676234, 0.0026657723133094993, 0.0016250492701108492, 0.0001797926852037535, 0.00021436820158909073, 0.0009611993555123746, 0.0004944298843103221, 0.0010580108013913188, 0.0002247408565046919, 0.0004356515064552489, 0.0007779491186700873, 0.022709199161889484, 0.0016872851996044562, 0.006773343659887561, 0.0003008069925524338, 0.005950446369916535, 0.00021436820158909073, 0.010037272406663394, 0.000397618438431378, 0.0005670384687195304, 0.007475226642509906, 0.0009266238391270374, 0.0021747999806377106, 0.0016181341668337816, 0.0001763351335652198, 0.0006119866400204687, 0.023224374356031006, 0.0003042645441909675, 0.0003215523023836361, 0.0021644273257221095, 0.00596427657647067, 0.000632731949851671, 0.0008298123932480931, 0.0012758365546189432, 0.0029631217542233994, 0.0002766041310826977, 0.010054560164856062, 0.00018325023684228725, 0.0003388400605763047, 0.0015386104791475061, 0.004318481996528618, 0.00522090297418592, 0.0011686524538243978, 0.0022543236683239863, 0.0007018829826223455, 0.0005359205039727268, 0.0003388400605763047, 0.01349136649355858, 0.00020053799503495586, 0.001341530035751084, 0.003858627628603633, 0.004128316656409264, 0.007257400889282281, 0.0007883217735856885, 0.0038378823187724307, 0.0010510956981142512, 0.0006050715367434012, 0.0002627739245285628, 0.0009646569071509083, 0.0030841360615720794, 0.00041490619662404657, 0.0004494817130093838, 0.000881675667826099, 0.00516558214796938, 0.0012308883833180048, 0.002295814287986391, 0.007385330299908029, 0.0006050715367434012, 0.004778336364453603, 0.0002904343376368326, 0.0010130626300903804, 0.000757203808838885, 0.0017598937840136643, 0.00025240126961296165, 0.0007260858440920815, 0.0009957748718977118, 0.0008228972899710258, 0.005870922682230259, 0.0019915497437954237, 0.0005290054006956594, 0.0005255478490571256, 0.006880527760682106, 0.0014314263783529607, 0.011067622794946443, 0.014822523874394065, 0.00019016534011935467, 0.01737419698363195, 0.0018117570585916701, 0.051586670446923125, 0.011686524538243979, 0.00042527885153964773, 0.00941145556008879, 0.0006189017432975362, 0.0019327713659403503, 0.0009162511842114362, 0.00034921271549190585, 0.021454107917101742, 0.018072622414615762, 0.00019708044339642212, 0.0006742225695140757, 0.0011340769374390608, 0.0013449875873896176, 0.06164123061177919, 0.027615464936968835, 0.00021782575322762445, 0.012084142976675356, 0.0024756069731901448, 0.007167504546680405, 0.002205917945384514, 0.006237423155914834, 0.0007018829826223455, 0.0008436425998022281, 0.00025240126961296165, 0.0015351529275089723, 0.002333847356010262, 0.0006915103277067443, 0.0005532082621653954, 0.006897815518874775, 0.005940073715000933, 0.0005877837785507327, 0.0035336177745814634, 0.00022128330486615819, 0.0003526702671304396, 0.00022128330486615819, 0.00032500985402216984, 0.0006534772596828734, 0.0010891287661381223, 0.0009854022169821107, 0.00585363492403759, 0.00025240126961296165, 0.0004010759900699117, 0.0002766041310826977, 0.001030350388283049, 0.0008436425998022281, 0.0002247408565046919, 0.002143682015890907, 0.0005048025392259233, 0.0032328107820290297, 0.002126394257698239, 0.001739148474182462, 0.00021436820158909073, 0.000670765017875542, 0.00031809475074510236, 0.0039900145908679146, 0.015068010040729959, 0.00023511351142029306, 0.004916638429994952, 0.011143688930994185, 0.0009646569071509083, 0.029703826126643202, 0.040698840337180436, 0.00018670778848082096, 0.000546293158888328, 0.006047257815795479, 0.0011617373505473305, 0.0028835980665371237, 0.0012723790029804095, 0.0027729564141040445, 0.026332713279072825, 0.0016803700963273886, 0.00025240126961296165, 0.000608529088381935, 0.0004183637482625803, 0.00822205779643319, 0.00029389188927536635, 0.006116408848566153, 0.004287364031781814, 0.0027210931395260387, 0.0029492915476692643, 0.0005981564334663339, 0.00035612781876897333, 0.00026623147616709655, 0.0003665004736845745, 0.002413371043696538, 0.0001936228917578884, 0.003402230812317182, 0.0005082600908644571, 0.005466389140521814, 0.0002696890278056303, 0.01300730926416386, 0.000359585370407507, 0.0009888597686206443, 0.00021436820158909073, 0.002627739245285628, 0.0006119866400204687, 0.00044602416137085006, 0.003907033351543105, 0.001925856262663283, 0.0004944298843103221, 0.000656934811321407, 0.0007917793252242223, 0.0001763351335652198, 0.00025240126961296165, 0.0015558982373401746, 0.0008851332194646327, 0.002904343376368326, 0.0018394174716999398, 0.0037410708728934866, 0.0010026899751747792, 0.016675771552648138, 0.0003665004736845745, 0.017294673295945674, 0.0023200171494561273, 0.005777568787989849, 0.00019708044339642212, 0.000992317320259178, 0.0017771815422063329, 0.0027003478296948363, 0.0004356515064552489, 0.003360740192654777, 0.008142534108746914, 0.0005704960203580641, 0.0006154441916590024, 0.0033296222279079737, 0.0028213621370435167, 0.00021091064995055702, 0.001279294106257477, 0.0006742225695140757, 0.001054553249752785, 0.003699580253231082, 0.0009439115973197059, 0.0003215523023836361, 0.01899233115046573, 0.0005981564334663339, 0.003806764354025627, 0.0007364584990076827, 0.0034471789836181205, 0.0005566658138039291, 0.00037687312860017566, 0.0002627739245285628, 0.0004114486449855129, 0.0013795631037749549, 0.0003077220958295012, 0.001912026056109148, 0.00023165595978175935, 0.017007696509947375, 0.003388400605763047, 0.0029458339960307307, 0.0009335389424041048, 0.0010752985595839874, 0.0022819840814322563, 0.0008713030129104978, 0.0002454861663358942, 0.0019535166757715526, 0.0002420286146973605, 0.0004356515064552489, 0.0007399160506462164, 0.00023165595978175935, 0.004408378339130495, 0.016402624973203973, 0.007779491186700873, 0.0002904343376368326, 0.005608148757701696, 0.0019293138143018165, 0.006489824425527795, 0.0009024209776573013, 0.00036304292204604075, 0.0008090670834168908, 0.001054553249752785, 0.011292363651451134, 0.0006915103277067443, 0.0002593163728900291, 0.0006915103277067443, 0.0003284674056607035, 0.006935848586898646, 0.0021505971191679747]



if __name__ == "__main__":
    print(sum(ATTRIBUTE_WEIGHTS))
    # recalculate_weights()
