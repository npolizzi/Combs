import numpy as np
from collections import Counter, defaultdict
import os
import pandas as pd
import pickle
import prody as pr
import itertools
from scipy.optimize import fsolve
import traceback


one_letter_code = {
    'CYS': 'C',
    'ASP': 'D',
    'SER': 'S',
    'GLN': 'Q',
    'LYS': 'K',
    'ILE': 'I',
    'PRO': 'P',
    'THR': 'T',
    'PHE': 'F',
    'ASN': 'N',
    'GLY': 'G',
    'HIS': 'H',
    'LEU': 'L',
    'ARG': 'R',
    'TRP': 'W',
    'ALA': 'A',
    'VAL': 'V',
    'GLU': 'E',
    'TYR': 'Y',
    'MET': 'M',
    'MSE': 'M',
    'HSE': 'H'}


# from https://web.expasy.org/protscale/pscale/A.A.Swiss-Prot.html
aa_bkgrd_freq = {
    'A': 0.0825,
    'C': 0.0137,
    'D': 0.0545,
    'E': 0.0675,
    'F': 0.0386,
    'G': 0.0707,
    'H': 0.0227,
    'I': 0.0596,
    'K': 0.0584,
    'L': 0.0966,
    'M': 0.0242,
    'N': 0.0406,
    'P': 0.047,
    'Q': 0.0393,
    'R': 0.0553,
    'S': 0.0656,
    'T': 0.0534,
    'V': 0.0687,
    'W': 0.0108,
    'Y': 0.0292}


# from terms(master) pds database
aa_bkgrd_freq_db = {
    'A': 0.07326110230321706,
    'C': 0.012867293906302065,
    'D': 0.058569031108557976,
    'E': 0.06983578684098528,
    'F': 0.0439535959408521,
    'G': 0.062122633180292236,
    'H': 0.024449412138684755,
    'I': 0.05847265868014285,
    'K': 0.06047821300742236,
    'L': 0.09626618778598098,
    'M': 0.02184252745196968,
    'N': 0.04386772159832098,
    'P': 0.04395611548146426,
    'Q': 0.040124104172086304,
    'R': 0.055522906508456316,
    'S': 0.05780687007337952,
    'T': 0.053235373594332544,
    'V': 0.06821782184454309,
    'W': 0.016032676762045978,
    'Y': 0.03911796762096367}


# from 20180207/pdbs_molprobity_biolassem
aa_bkgrd_freq_db_biolassm = {
 'A': 0.05527016560167656,
 'C': 0.008441566240140275,
 'D': 0.04617196010428079,
 'E': 0.06373058485785288,
 'F': 0.052315870616027944,
 'G': 0.03386366080474732,
 'H': 0.02663216051492069,
 'I': 0.07046184722672305,
 'K': 0.07511650633562277,
 'L': 0.11354422228109655,
 'M': 0.01912097864055931,
 'N': 0.03825583186914515,
 'P': 0.04164194988361941,
 'Q': 0.03980624437210954,
 'R': 0.07844738261799139,
 'S': 0.04207770775245258,
 'T': 0.05016927852792366,
 'V': 0.07400908838550538,
 'W': 0.022363936910759847,
 'Y': 0.04855905645684492}

## from biounits/20180719/opdb
## To make the bkgrd_freq:
# import os
# from multiprocessing import Pool
# import prody as pr
# from collections import defaultdict
# def func(f):
#     counts = defaultdict(int)
#     pdb = pr.parsePDB(f)
#     aas = set('GALMFWKQESPVICYHRNDT')
#     for aa in aas:
#         sel = pdb.select('name CA and sequence ' + aa)
#         if sel is not None:
#             counts[aa] += len(sel)
#     return counts
# it = os.listdir()  # opdb dir
# with Pool(16) as p:
#     results = p.map(func, it, 1250)
# counts = defaultdict(int)
# for result in results:
#     for key, val in result.items():
#         counts[key] += val
# tot = np.sum(list(counts.values()))
# freq = dict()
# for key, val in counts.items():
#     freq[key] = val/tot

aa_bkgrd_freq_biounits = {
    'S': 0.05914184683030594,
     'W': 0.013147493114676684,
     'T': 0.054170068063253272,
     'L': 0.095612697659689716,
     'E': 0.068663311935271379,
     'K': 0.056168044682294277,
     'C': 0.012967896657862093,
     'A': 0.085720084657585502,
     'G': 0.073390013540377344,
     'Y': 0.034175493995541609,
     'H': 0.023691924422224477,
     'R': 0.051818468466081881,
     'N': 0.041140358706563164,
     'D': 0.058414766469586044,
     'F': 0.040242376422490206,
     'V': 0.072892523203875806,
     'I': 0.059574807426272733,
     'P': 0.044970708252620215,
     'Q': 0.036600589570879943,
     'M': 0.017496525922547738}


aa_bkgrd_freq_biounits_buried_binary = {
     'D': 0.031640186203678325,
     'W': 0.018573067295938987,
     'G': 0.071011105668014968,
     'I': 0.089053355834742443,
     'Y': 0.046530450342596952,
     'T': 0.049549386323948924,
     'P': 0.03365032068933755,
     'A': 0.10242419397535006,
     'E': 0.02979795479119856,
     'V': 0.10426033993871953,
     'R': 0.034100643923499263,
     'N': 0.02850175413070604,
     'Q': 0.022635104577059903,
     'L': 0.1380403913386356,
     'C': 0.020597308959081163,
     'H': 0.023009912919989346,
     'S': 0.050913356758624254,
     'K': 0.022129182466935956,
     'F': 0.060016082182330566,
     'M': 0.023565901679611617}


aa_bkgrd_freq_biounits_exposed_binary = {
     'D': 0.084257836611558987,
     'W': 0.0079120984830468397,
     'G': 0.075725275694279295,
     'I': 0.031120920699984234,
     'Y': 0.022253278002976724,
     'T': 0.058640380293771062,
     'P': 0.055917239610193542,
     'A': 0.069575696684186755,
     'E': 0.10616908536569072,
     'V': 0.04260936251626643,
     'R': 0.068923970638926663,
     'N': 0.053339995778076459,
     'Q': 0.050076555765464131,
     'L': 0.054667764010506714,
     'C': 0.005600728949836601,
     'H': 0.024342862945139043,
     'S': 0.067071401277799897,
     'K': 0.08901471527401085,
     'F': 0.021156913934218157,
     'M': 0.011623917464066888}


# The following is derived from the summed counts from buried and exposed.
# There were some errors with residues not being found in the on_surface_dict,
# so these freqs are slightly different than aa_bkgrd_freq_biounits.
# Total number of error-free residues in the database was 7357551. Total buried was
# 3615181. Total surface-exposed was 3742370.

aa_bkgrd_freq_biounits_all_binary = {
     'D': 0.058403808549882974,
     'W': 0.013150435518557737,
     'G': 0.073408937294488344,
     'I': 0.059586403138761797,
     'Y': 0.034182026057311736,
     'T': 0.054173460707238046,
     'P': 0.044976242774260079,
     'A': 0.085716021540319598,
     'E': 0.068643628837910878,
     'V': 0.072901975127321578,
     'R': 0.051813300376715027,
     'N': 0.041135562634903922,
     'Q': 0.036593018519341561,
     'L': 0.095633451946170678,
     'C': 0.012969397018111053,
     'H': 0.023687909196959695,
     'S': 0.059132039995373462,
     'K': 0.056150069500027933,
     'F': 0.040250621436399146,
     'M': 0.017491689829944774}


## The following frequency distributions were derived by the following code:

# from multiprocessing import Pool
# import os
# import prody as pr
# import pickle
# import numpy as np
# from collections import defaultdict
#
# path_to_opdb = 'opdb/'
# with open('distance_dict_20180801.pkl', 'rb') as infile:
#     db = pickle.load(infile)
#
# a2aaa = {
# 'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN',
# 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',
# 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP',
# 'Y': 'TYR', 'V': 'VAL'
# }
# aaa2a = {aaa:a for a, aaa in a2aaa.items()}
# # add unnatural amino acids (the most common ones)
# aaa2a['ASX'] = aaa2a['ASN']
# aaa2a['CSO'] = aaa2a['CYS']
# aaa2a['GLX'] = aaa2a['GLU'] # or GLN
# aaa2a['HIP'] = aaa2a['HIS']
# aaa2a['HSC'] = aaa2a['HIS']
# aaa2a['HSD'] = aaa2a['HIS']
# aaa2a['HSE'] = aaa2a['HIS']
# aaa2a['HSP'] = aaa2a['HIS']
# aaa2a['MSE'] = aaa2a['MET']
# aaa2a['SEC'] = aaa2a['CYS']
# aaa2a['SEP'] = aaa2a['SER']
# aaa2a['TPO'] = aaa2a['THR']
# aaa2a['PTR'] = aaa2a['TYR']
# aaa2a['XLE'] = aaa2a['LEU'] # or ILE
#
# def func(acc):
#     surf_dict = dict()
#     int_dict = dict()
#     buried_dict = dict()
#     for a_ in aaa2a.values():
#         surf_dict[a_] = 0
#         int_dict[a_] = 0
#         buried_dict[a_] = 0
#     opdb = pr.parsePDB(path_to_opdb + acc + '.pdb')
#     for ri, dist in db[acc].items():
#         try:
#             dist = float(dist)
#             sel = opdb.select('resindex ' + str(ri))
#             aaa = set(sel.getResnames()).pop()
#             a = aaa2a[aaa]
#             if dist <= 1:
#                 surf_dict[a] += 1
#             elif (dist > 1) and (dist <= 3):
#                 int_dict[a] += 1
#             else:
#                 buried_dict[a] += 1
#         except:
#             print(acc, ri, dist)
#     return surf_dict, int_dict, buried_dict
#
# it = db.keys()
# with Pool(8) as p:
#     results = p.map(func, it, 2500)
#
# counts_exp = defaultdict(int)
# counts_int = defaultdict(int)
# counts_bur = defaultdict(int)
# for (result_exp, result_int, result_bur) in results:
#     for key, val in result_exp.items():
#         counts_exp[key] += val
#     for key, val in result_int.items():
#         counts_int[key] += val
#     for key, val in result_bur.items():
#         counts_bur[key] += val
# freq_exp = dict()
# tot_exp = np.sum(list(counts_exp.values()))
# for key, val in counts_exp.items():
#     freq_exp[key] = val/tot_exp
# freq_bur = dict()
# tot_bur = np.sum(list(counts_bur.values()))
# for key, val in counts_bur.items():
#     freq_bur[key] = val/tot_bur
# freq_int = dict()
# tot_int = np.sum(list(counts_int.values()))
# for key, val in counts_int.items():
#     freq_int[key] = val/tot_int


# total counts = 3755649
aa_bkgrd_freq_biounits_exposed_ternary = {
     'A': 0.06885068333063074,
     'R': 0.0691457055757873,
     'N': 0.05355851944630608,
     'D': 0.08475925199612637,
     'C': 0.005313063068460338,
     'Q': 0.050234992673703,
     'E': 0.10706298698307536,
     'G': 0.0754295728913964,
     'H': 0.024268774850897942,
     'I': 0.030223271663566004,
     'L': 0.05329891052119088,
     'K': 0.08980152298577423,
     'M': 0.014584430014625968,
     'F': 0.02055303890219773,
     'P': 0.056042777160485446,
     'S': 0.0671220872877098,
     'T': 0.058679871308527504,
     'W': 0.007747795387694644,
     'Y': 0.021734459210645086,
     'V': 0.04158828474119919}


# total counts = 454440
aa_bkgrd_freq_biounits_intermediate_ternary = {
     'A': 0.10883284922101928,
     'R': 0.03836149986796937,
     'N': 0.034215738051227886,
     'D': 0.03987325059413784,
     'C': 0.019978434996919286,
     'Q': 0.02682862424082387,
     'E': 0.03797641052724232,
     'G': 0.10590176921045683,
     'H': 0.026388522137135814,
     'I': 0.06077590000880204,
     'L': 0.10678417392835138,
     'K': 0.030248217586480063,
     'M': 0.020275503916908723,
     'F': 0.04825499515887686,
     'P': 0.03850233254114955,
     'S': 0.058276120059853886,
     'T': 0.04977554792711909,
     'W': 0.0177009066103336,
     'Y': 0.05086039961271015,
     'V': 0.08018880380248218}


# total counts = 3183357
aa_bkgrd_freq_biounits_buried_ternary = {
     'A': 0.10136374902343658,
     'R': 0.032708866771775834,
     'N': 0.02701016568358497,
     'D': 0.029306483690016545,
     'C': 0.020858797803702193,
     'Q': 0.02149146325718416,
     'E': 0.02693917144699762,
     'G': 0.06559333433227879,
     'H': 0.02235784425058201,
     'I': 0.09339794437130362,
     'L': 0.14292270706678517,
     'K': 0.019527184667004045,
     'M': 0.03136406001588889,
     'F': 0.06190194816352674,
     'P': 0.03234855531440552,
     'S': 0.049177330723509806,
     'T': 0.04888864177030726,
     'W': 0.01872991310745229,
     'Y': 0.046109500128323655,
     'V': 0.10800233841193432}


aa_bkgrd_freq_biounits_all_ternary = {
     'A': 0.08530717611246501,
     'R': 0.05156512944031782,
     'N': 0.04093882609002622,
     'D': 0.05812431713168663,
     'C': 0.01290791871611695,
     'Q': 0.0364203647392569,
     'E': 0.06831807522500333,
     'G': 0.07306741673639058,
     'H': 0.023576286348747255,
     'I': 0.05930198178224336,
     'L': 0.09517524034124276,
     'K': 0.05588341349892865,
     'M': 0.022158949967308883,
     'F': 0.04005912804394595,
     'P': 0.044762753389961864,
     'S': 0.05885198863966816,
     'T': 0.05391680685839864,
     'W': 0.01308807827905959,
     'Y': 0.03401972503755353,
     'V': 0.0725564236216779}



#intermediate frequencies by secondary structure:
aa_bkgrd_freq_biounits_intermediate_ternary_ss = {
             '-': {'A': 0.08447348906192065,
              'C': 0.025259547645532072,
              'D': 0.056970708194289954,
              'E': 0.02267334074898035,
              'F': 0.04793288839451242,
              'G': 0.12869855394883203,
              'H': 0.02442528735632184,
              'I': 0.05253985910270671,
              'K': 0.0232202447163515,
              'L': 0.09646829810901002,
              'M': 0.020281794586577678,
              'N': 0.03993325917686318,
              'P': 0.07898591027067112,
              'Q': 0.02004078605858361,
              'R': 0.029912866147571375,
              'S': 0.07123655913978495,
              'T': 0.05380978865406007,
              'V': 0.06763070077864293,
              'W': 0.014923989618094179,
              'Y': 0.04058212829069336},
             'B': {'A': 0.053889255108767305,
              'C': 0.026697429136453527,
              'D': 0.037574159525379035,
              'E': 0.023895847066578775,
              'F': 0.06114040870138431,
              'G': 0.08899143045484509,
              'H': 0.025214238628872777,
              'I': 0.06905075807514832,
              'K': 0.031147000659195782,
              'L': 0.11189848384970336,
              'M': 0.02290705339485827,
              'N': 0.03213579433091628,
              'P': 0.03444297956493078,
              'Q': 0.022577455504284773,
              'R': 0.044001318391562294,
              'S': 0.06822676334871457,
              'T': 0.07218193803559657,
              'V': 0.09212261041529335,
              'W': 0.02010547132498352,
              'Y': 0.061799604482531315},
             'E': {'A': 0.06848511024729503,
              'C': 0.01907040101000033,
              'D': 0.02974628171478565,
              'E': 0.041219531102915936,
              'F': 0.05145242920584294,
              'G': 0.0922954250971793,
              'H': 0.027575666965679924,
              'I': 0.07784311771155188,
              'K': 0.043877426714065804,
              'L': 0.08572820802462983,
              'M': 0.018560971017863272,
              'N': 0.02633531568047665,
              'P': 0.016191014097921303,
              'Q': 0.028229066303420935,
              'R': 0.049724797058595525,
              'S': 0.0548744698051984,
              'T': 0.06156350709325891,
              'V': 0.12824346323798133,
              'W': 0.01827303232665537,
              'Y': 0.060710765584681664},
             'G': {'A': 0.0992498880429915,
              'C': 0.01964845499328258,
              'D': 0.05648231079265562,
              'E': 0.04064039408866995,
              'F': 0.06924540976265114,
              'G': 0.07753022839229735,
              'H': 0.03638602776533811,
              'I': 0.03912897447380206,
              'K': 0.029948499776085984,
              'L': 0.11587550380653829,
              'M': 0.01763322884012539,
              'N': 0.042599641737572774,
              'P': 0.054858934169279,
              'Q': 0.03000447828034035,
              'R': 0.03767353336318854,
              'S': 0.05625839677563815,
              'T': 0.03011643528884908,
              'V': 0.04148007165248545,
              'W': 0.027765338110165697,
              'Y': 0.077474249888043},
             'H': {'A': 0.16492153483905764,
              'C': 0.015686455058470685,
              'D': 0.03036003209639442,
              'E': 0.05238026334828534,
              'F': 0.04283027058312835,
              'G': 0.06058194446271425,
              'H': 0.023618473842753975,
              'I': 0.07060549058812697,
              'K': 0.030432380526433486,
              'L': 0.13245682114152668,
              'M': 0.02242801331211113,
              'N': 0.026354559924231462,
              'P': 0.018718511990107996,
              'Q': 0.03391825942831586,
              'R': 0.039962641901579825,
              'S': 0.0478157351258205,
              'T': 0.041054445482169404,
              'V': 0.07880059457255233,
              'W': 0.018508043829994342,
              'Y': 0.04856552794622539},
             'I': {'A': 0.0762749445676275,
              'C': 0.02483370288248337,
              'D': 0.043458980044345896,
              'E': 0.04478935698447894,
              'F': 0.0762749445676275,
              'G': 0.07804878048780488,
              'H': 0.03680709534368071,
              'I': 0.08381374722838138,
              'K': 0.03725055432372506,
              'L': 0.12106430155210643,
              'M': 0.015521064301552107,
              'N': 0.040354767184035474,
              'P': 0.003547671840354767,
              'Q': 0.022172949002217297,
              'R': 0.029711751662971176,
              'S': 0.026607538802660754,
              'T': 0.05232815964523282,
              'V': 0.08957871396895788,
              'W': 0.01818181818181818,
              'Y': 0.07937915742793791},
             'S': {'A': 0.0733552054897443,
              'C': 0.022990808685416614,
              'D': 0.04826066267625034,
              'E': 0.02421798692679506,
              'F': 0.04946279646372311,
              'G': 0.17556162187883492,
              'H': 0.028801121991534975,
              'I': 0.04528037266147412,
              'K': 0.023942497933832553,
              'L': 0.09947657091337124,
              'M': 0.019810163039394927,
              'N': 0.04327681634901951,
              'P': 0.03183150091412257,
              'Q': 0.019384407322998322,
              'R': 0.03290841243206692,
              'S': 0.07185253825540334,
              'T': 0.060407222820506395,
              'V': 0.0658669137719452,
              'W': 0.013774449648125422,
              'Y': 0.049537929825440155},
             'T': {'A': 0.10230246636178593,
              'C': 0.020090406830738324,
              'D': 0.03706151365354622,
              'E': 0.031272304316793995,
              'F': 0.047979063681302704,
              'G': 0.19921224457426842,
              'H': 0.03209178143752148,
              'I': 0.028100134817203734,
              'K': 0.023394750059478178,
              'L': 0.08530492479314811,
              'M': 0.017341193264426763,
              'N': 0.054640619630442255,
              'P': 0.05786565862169235,
              'Q': 0.021385709376404344,
              'R': 0.03473525602051336,
              'S': 0.05815644082582146,
              'T': 0.03965211874487827,
              'V': 0.038039599249253216,
              'W': 0.020063972084908403,
              'Y': 0.051309841655872476}}


#exposed frequencies by secondary structure:
aa_bkgrd_freq_biounits_exposed_ternary_ss = {
              '-': {'A': 0.054137896652292045,
              'C': 0.007281202482737992,
              'D': 0.10124760076775433,
              'E': 0.06910038138445049,
              'F': 0.019647530971907173,
              'G': 0.0706097165790064,
              'H': 0.025171373731834385,
              'I': 0.02772640027918339,
              'K': 0.07515267841563426,
              'L': 0.04672458060173991,
              'M': 0.019526634593813097,
              'N': 0.06122092878330882,
              'P': 0.10217114938803998,
              'Q': 0.039349901538001344,
              'R': 0.05565844903656804,
              'S': 0.08489916992796072,
              'T': 0.0732494952264626,
              'V': 0.04027469651270035,
              'W': 0.0069608893985093605,
              'Y': 0.01988932372809532},
             'B': {'A': 0.047556061398381984,
              'C': 0.006890399581654311,
              'D': 0.06232120335906979,
              'E': 0.06118305699960011,
              'F': 0.032421790888676985,
              'G': 0.04555661509120551,
              'H': 0.02562367344427697,
              'I': 0.046756282875511396,
              'K': 0.09154388015626441,
              'L': 0.06533575317604355,
              'M': 0.01476514196068781,
              'N': 0.05029376480359285,
              'P': 0.05899904641791504,
              'Q': 0.042019133163124056,
              'R': 0.07465624903872774,
              'S': 0.0692116029407241,
              'T': 0.08837552677720016,
              'V': 0.07194930634593497,
              'W': 0.01233504567965794,
              'Y': 0.03220646590175028},
             'E': {'A': 0.04057685631310858,
              'C': 0.007338367631094106,
              'D': 0.05214557704918635,
              'E': 0.09625109524448061,
              'F': 0.03229249135284966,
              'G': 0.03578074169864383,
              'H': 0.027167573783199454,
              'I': 0.05892920449890613,
              'K': 0.0933433015123099,
              'L': 0.05883001254594516,
              'M': 0.013690326396631883,
              'N': 0.036111381541847065,
              'P': 0.02752209317063404,
              'Q': 0.04622712385584837,
              'R': 0.07655598191767435,
              'S': 0.06466213200244673,
              'T': 0.09493404653572105,
              'V': 0.0921291185325469,
              'W': 0.011561373184006584,
              'Y': 0.03395120123291924},
             'G': {'A': 0.08181545450649295,
              'C': 0.004255162828856739,
              'D': 0.1049463971934293,
              'E': 0.13175086174700454,
              'F': 0.02152071560205962,
              'G': 0.0499476523134004,
              'H': 0.026271803882912614,
              'I': 0.01744310633009043,
              'K': 0.0856787750029082,
              'L': 0.05180278085605305,
              'M': 0.011669554463022942,
              'N': 0.05001500021428878,
              'P': 0.07977052733406395,
              'Q': 0.047388432079641954,
              'R': 0.059945754327102634,
              'S': 0.0863644990846808,
              'T': 0.0374638005032725,
              'V': 0.020761521082954246,
              'W': 0.00900625110970973,
              'Y': 0.022181949538054625},
             'H': {'A': 0.09967200334031406,
              'C': 0.0034683485669922166,
              'D': 0.074096908252272,
              'E': 0.1538069468094129,
              'F': 0.018901806509698003,
              'G': 0.02216546297927309,
              'H': 0.020996026852993562,
              'I': 0.032404145381950564,
              'K': 0.1084403277683179,
              'L': 0.06996636851753991,
              'M': 0.016147026011390994,
              'N': 0.03828476177428089,
              'P': 0.025949412509215224,
              'Q': 0.06848622446649574,
              'R': 0.08957929657683049,
              'S': 0.0505883062911423,
              'T': 0.04159653311934446,
              'V': 0.03658198448580693,
              'W': 0.008036815219305972,
              'Y': 0.020831294567422803},
             'I': {'A': 0.04155895804334858,
              'C': 0.005700271757141911,
              'D': 0.0946510240604494,
              'E': 0.1492012991317028,
              'F': 0.033008550407635714,
              'G': 0.03148405912374892,
              'H': 0.036786637502485585,
              'I': 0.03711804865115662,
              'K': 0.10134552926360443,
              'L': 0.0561410485848744,
              'M': 0.011267979054815405,
              'N': 0.0609796513554716,
              'P': 0.007423609730231325,
              'Q': 0.0473917942599589,
              'R': 0.07085570358586862,
              'S': 0.04507191621926162,
              'T': 0.05799695101743223,
              'V': 0.0657519718963346,
              'W': 0.008417843176244449,
              'Y': 0.037847153178232916},
             'S': {'A': 0.051228683771503976,
              'C': 0.005399127771310864,
              'D': 0.10558204904006566,
              'E': 0.08141069311604504,
              'F': 0.017777157907724,
              'G': 0.14528562002800144,
              'H': 0.025198611729491092,
              'I': 0.02017230003379484,
              'K': 0.07810094464620022,
              'L': 0.03617388785477875,
              'M': 0.010181365633331366,
              'N': 0.06727854993321496,
              'P': 0.05722592654182245,
              'Q': 0.039400490293371386,
              'R': 0.055251878832093294,
              'S': 0.0815367532279435,
              'T': 0.06802418208444418,
              'V': 0.030755985173185136,
              'W': 0.005216742928564148,
              'Y': 0.018799049453113684},
             'T': {'A': 0.06068438111958944,
              'C': 0.004792693632250659,
              'D': 0.09741542305354851,
              'E': 0.0831335674252144,
              'F': 0.01457728408400297,
              'G': 0.19381963670740765,
              'H': 0.024314605982848268,
              'I': 0.011648321966371802,
              'K': 0.07668647444121818,
              'L': 0.03291410628671754,
              'M': 0.009041798906070632,
              'N': 0.08281956918090351,
              'P': 0.07603146735093524,
              'Q': 0.03885475048956716,
              'R': 0.049625227901951514,
              'S': 0.0649858194341279,
              'T': 0.03978323992166925,
              'V': 0.017788169356472417,
              'W': 0.005746505503410089,
              'Y': 0.015336957255722872}}


#buried frequencies by secondary structure:
aa_bkgrd_freq_biounits_buried_ternary_ss = {
             '-': {'A': 0.08309958403360525,
              'C': 0.02624043287051948,
              'D': 0.04894973853262047,
              'E': 0.022344285878809527,
              'F': 0.05108715384119883,
              'G': 0.08252765464053359,
              'H': 0.02740185945371959,
              'I': 0.06771604974419336,
              'K': 0.017554133214653104,
              'L': 0.1056175959055321,
              'M': 0.026025715350936268,
              'N': 0.04028881458361393,
              'P': 0.09166876504242623,
              'Q': 0.019482686936000516,
              'R': 0.0300975403132143,
              'S': 0.06908438593717366,
              'T': 0.05990033203136438,
              'V': 0.07854171668608884,
              'W': 0.014653494722828961,
              'Y': 0.037718060280967636},
             'B': {'A': 0.05748537589369539,
              'C': 0.028309381093377627,
              'D': 0.03232950578946101,
              'E': 0.019522881009123517,
              'F': 0.07638236922558436,
              'G': 0.06489973760862762,
              'H': 0.029224140006258878,
              'I': 0.10180303796249489,
              'K': 0.017524854962567102,
              'L': 0.11595772850918369,
              'M': 0.03059627837558075,
              'N': 0.029850027683493416,
              'P': 0.03290724826075443,
              'Q': 0.0218338508942972,
              'R': 0.03748104282516068,
              'S': 0.05820755398281216,
              'T': 0.06246840470860114,
              'V': 0.10623239690907778,
              'W': 0.0213042536289449,
              'Y': 0.055679930670903446},
             'E': {'A': 0.07732629037248553,
              'C': 0.023222963503738203,
              'D': 0.020148072050136606,
              'E': 0.02057477517194518,
              'F': 0.0705839765899747,
              'G': 0.050998101069993466,
              'H': 0.020470627253589058,
              'I': 0.12736987071502096,
              'K': 0.01776985939019877,
              'L': 0.13332955839260327,
              'M': 0.027057224730984916,
              'N': 0.019212763074608334,
              'P': 0.014483638665369706,
              'Q': 0.017141938445546817,
              'R': 0.027912653264278883,
              'S': 0.04269559080181763,
              'T': 0.04936914673531666,
              'V': 0.16574281733264037,
              'W': 0.020025723524689124,
              'Y': 0.05456440891506181},
             'G': {'A': 0.0962044305618347,
              'C': 0.018577123833792283,
              'D': 0.04392544490567589,
              'E': 0.033732686695984546,
              'F': 0.07350704862110065,
              'G': 0.06946899017714027,
              'H': 0.028995931116682423,
              'I': 0.05173441288890716,
              'K': 0.020827339607907606,
              'L': 0.1486375405860836,
              'M': 0.030670749249928075,
              'N': 0.03576712835477375,
              'P': 0.048446426369651886,
              'Q': 0.02596481854424397,
              'R': 0.03423615963174551,
              'S': 0.060314002712588875,
              'T': 0.03962023755702602,
              'V': 0.05251530968723028,
              'W': 0.029982327072459004,
              'Y': 0.05687189182524351},
             'H': {'A': 0.1398287344730976,
              'C': 0.01688928090314583,
              'D': 0.022363725614091192,
              'E': 0.033832721670737254,
              'F': 0.05911678156295053,
              'G': 0.042144111662166236,
              'H': 0.0187779987156375,
              'I': 0.09307587625072751,
              'K': 0.021801494640322622,
              'L': 0.18009358480704044,
              'M': 0.039385397296992926,
              'N': 0.020710560227918465,
              'P': 0.013565756523039865,
              'Q': 0.02570702264312174,
              'R': 0.03674273978420303,
              'S': 0.04055112390315529,
              'T': 0.04240459481973792,
              'V': 0.09216117631786855,
              'W': 0.01863271273006122,
              'Y': 0.04221460545398432},
             'I': {'A': 0.075259363460524,
              'C': 0.016001406717074028,
              'D': 0.027694742394935818,
              'E': 0.027079303675048357,
              'F': 0.07649024090029893,
              'G': 0.05991735537190083,
              'H': 0.025101107789695796,
              'I': 0.10093195006154387,
              'K': 0.019474239493581853,
              'L': 0.15539827677158433,
              'M': 0.030552136451556182,
              'N': 0.04549850536310884,
              'P': 0.0055389484789871634,
              'Q': 0.020221557939159487,
              'R': 0.02334271144716019,
              'S': 0.036574643924740635,
              'T': 0.04769650079127835,
              'V': 0.12278002461754879,
              'W': 0.02167223492175136,
              'Y': 0.0627747494285212},
             'S': {'A': 0.07207385576177897,
              'C': 0.02064748242135572,
              'D': 0.053807904874502904,
              'E': 0.025065477975075943,
              'F': 0.053841845941496914,
              'G': 0.15208426435565714,
              'H': 0.029766315753746244,
              'I': 0.05512594964277027,
              'K': 0.020302414906916623,
              'L': 0.09668112933243578,
              'M': 0.024397970324193757,
              'N': 0.045362236037493564,
              'P': 0.047658914904088204,
              'Q': 0.020483433930884674,
              'R': 0.03435967348693552,
              'S': 0.06886076808634607,
              'T': 0.05863319323215124,
              'V': 0.06587395419087325,
              'W': 0.016110693133156464,
              'Y': 0.038862521708140765},
             'T': {'A': 0.08202555049396879,
              'C': 0.018550282347437712,
              'D': 0.036131561654210706,
              'E': 0.030289698495260794,
              'F': 0.05933194480024058,
              'G': 0.16308209794727288,
              'H': 0.028396244277869975,
              'I': 0.039194502299989976,
              'K': 0.019151732510608914,
              'L': 0.1085840303843712,
              'M': 0.025884632948330977,
              'N': 0.04699664747223862,
              'P': 0.06266219662965149,
              'Q': 0.022404018578127263,
              'R': 0.038125257565463394,
              'S': 0.0579842509160977,
              'T': 0.049318913380038534,
              'V': 0.04694095764231536,
              'W': 0.019313233017386366,
              'Y': 0.045632246639118766}}

# These functions below for PSSM are based on the PsiBlast paper:
# Nucleic Acids Research, 1997, Vol. 25, No. 17 3389–3402
def find_lambda(aa_bkgrd_freq, blossum):
    """Finds the optimum lambda for weighting the PSSM,
    see formula 3 of NucAcRes paper"""
    def eq(x):
        return np.sum(aa_bkgrd_freq[aa1] * aa_bkgrd_freq[aa2]
                      * np.exp(x * blossum[aa1][aa2]) for aa1, aa2 in
                      itertools.product(aa_bkgrd_freq.keys(), repeat=2)) - 1
    lambda_ = fsolve(eq, .3)
    return lambda_


def get_qs(lambda_, aa_bkgrd_freq, blossum):
    """Gets the target frequencies of amino acids,
    see formula 3 of NucAcRes paper"""
    qs = defaultdict(dict)
    for aa1, aa2 in itertools.product(aa_bkgrd_freq.keys(), repeat=2):
        qs[aa1][aa2] = aa_bkgrd_freq[aa1]*aa_bkgrd_freq[aa2] * np.exp(lambda_ * blossum[aa1][aa2])
    return qs


def get_gs(seq_dict, aa_bkgrd_freq, qs):
    """construct pseudocount frequencies, see formula 4 of NucAcRes paper"""
    return {aa_i: np.sum(freq_j / aa_bkgrd_freq[aa_j] * qs[aa_i][aa_j]
                         for aa_j, freq_j in seq_dict.items()) for aa_i in aa_bkgrd_freq.keys()}


def get_Qs(alpha, beta, seq_dict, gs):
    """Calculates the observed amino acid frequency with pseudocounts,
    see formula 5 of NucAcRes paper"""
    return {aa: (alpha * seq_dict[aa] + beta * gs[aa]) / (alpha + beta) for aa in gs.keys()}


def calc_Nc(seqfile):
    """Calculates the mean number of residue types across the columns
    of the sequence alignment see bottom of pg 3395 of NucAcRes paper"""
    seqs = list()
    with open(seqfile, 'r') as seqf:
        for line in seqf:
            seqs.append([one_letter_code[i] for i in line.split()[2:]])
    seqs = np.array(seqs)
    return np.mean([len(np.unique(seqs[:, i])) for i in range(seqs.shape[1])])


def get_alpha(seqfile):
    """Needed for calc of Q, see under formula 5 of NucAcRes paper"""
    return calc_Nc(seqfile) - 1


def calc_column_scores(lambda_, Qs, aa_bkgrd_freq):
    """Calcs the pssm scores for one column of pssm, see bottom of first
    paragraph, column 2 of pg 3396 of NucAcRes paper"""
    return {aa: np.log(Qs[aa] / aa_bkgrd_freq[aa]) / lambda_ for aa in Qs.keys()}


def make_pssm(output_dir):
    """Returns a pandas dataframe of the PSSM from results
    in TERMs output directory"""
    if output_dir[-1] != '/':
        output_dir += '/'
    frag_path = output_dir + 'fragments/'
    designscore_path = output_dir + 'designscore/'
    pssm = dict()
    beta = 10
    lambda_ = 0.31902034
    blossum = blossum62
    terms = [f.split('.')[0] for f in os.listdir(frag_path) if f[:3] != 'hit']
    for term in terms:
        try:
            rns = get_resnums(frag_path + term + '.pdb')
            rn = int(term.split('_')[-1][1:])
            seqfile = designscore_path + 't1k_' + term + '.seq'
            seq_dict = zero_order_freq(rn, rns, seqfile)
            qs = get_qs(lambda_, aa_bkgrd_freq_biounits, blossum)
            gs = get_gs(seq_dict, aa_bkgrd_freq_biounits, qs)
            alpha = get_alpha(seqfile)
            Qs = get_Qs(alpha, beta, seq_dict, gs)
            pssm[rn] = calc_column_scores(lambda_, Qs, aa_bkgrd_freq_biounits)
        except:
            print('Exception: ', term)
            traceback.print_exc()
    pssm = pd.DataFrame(pssm)
    return pssm


def print_pssm_rosetta(pssm, outdir, filetag=''):
    """Prints the PSSM matrix as a text file formatted
    to Rosetta specifications"""
    if outdir[-1] != '/':
        outdir += '/'
    pssm_ = pssm.transpose()
    pssm_[' '] = 'A'
    pssm_ = pssm_[[' ', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G',
                   'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T',
                   'W', 'Y', 'V']]
    with open(outdir + 'pssm' + filetag + '.txt', 'w') as outfile:
        outfile.write('\n')
        outfile.write('comment\n')
        outfile.write(pssm_.to_string())


def parse_dssp(dssp_file):
    """parses a DSSP file and returns a dictionary of
    keys=(resnum, chain) and values=(dssp code)"""
    dssp_ss = dict()
    with open(dssp_file, 'r') as infile:
        for line in infile:
            if line.startswith('  #  RESIDUE'):
                break
        for line in infile:
            if line[13] == '!':
                continue
            resnum = int(line[5:10])
            chain = line[11]
            if line[16] != ' ':
                dssp_ss[(resnum, chain)] = line[16]
            else:
                dssp_ss[(resnum, chain)] = '-'
    return dssp_ss


def parse_dssps(dssp_dir):
    """Returns a pandas Dataframe of (aa code, dssp code, phi, psi)
    of residues in proteins in dssp_dir."""
    if dssp_dir[-1] != '/':
        dssp_dir += '/'
    data = list()
    for dssp_file in [f for f in os.listdir(dssp_dir) if f[-4:] == 'dssp']:
        with open(dssp_dir + dssp_file, 'r') as infile:
            for line in infile:
                if line.startswith('  #  RESIDUE'):
                    break
            for line in infile:
                if line[13] == '!':
                    continue
                aa = line[13]
                if line[16] != ' ':
                    data.append((aa, line[16], float(line[103:109]), float(line[109:115])))
                else:
                    data.append((aa, '-', float(line[103:109]), float(line[109:115])))
    df = pd.DataFrame(data, columns=['aa', 'dssp', 'phi', 'psi'])
    df = df[df['aa'].isin(one_letter_code.values())]
    return df


def make_ss_propensity(df_dssp, save_pickle=True):
    """Takes the Dataframe of parsed DSSP info and outputs dictionary
    of ss propensities as pandas Dataframes."""
    aa_dssp_propensities = dict()
    for dssp_ in set(df_dssp['dssp']):
        cond = (df_dssp['dssp'] == dssp_)
        df_ss = df_dssp[cond]
        df_not_ss = df_dssp[~cond]
        aa_dssp_propensities[dssp_] = (df_ss['aa'].value_counts()/len(df_ss)) / (
            df_not_ss['aa'].value_counts()/len(df_not_ss))
    if save_pickle:
        with open('aa_dssp_propensities.pkl', 'wb') as outfile:
            pickle.dump(aa_dssp_propensities, outfile)
    return aa_dssp_propensities


def get_resnums(pdbfile):
    """Returns the resnums of the TERM pdb fragment"""
    resnums = list()
    with open(pdbfile, 'r') as pdb:
        pdb.readline()
        for line in pdb:
            resnums.append(int(line.split()[5]))
    return list(np.unique(resnums))


def zero_order_freq_nonzero(resnum, resnums, seqfile):
    """Returns a dictionary of the AA freq at resnum from the TERM seqfile"""
    index = resnums.index(resnum) + 2  # plus 2 because sequence info starts at column 3.
    seq_distrib = list()
    with open(seqfile, 'r') as seqs:
        for line in seqs:
            seq_distrib.append(one_letter_code[line.split()[index]])
    total_count = len(seq_distrib)
    seq_distrib_cnt = Counter(seq_distrib)
    seq_dict = {aa: count/total_count for aa, count in seq_distrib_cnt.items()}
    return seq_dict


def zero_order_freq(resnum, resnums, seqfile):
    """Returns a dictionary of the AA freq at resnum from the TERM seqfile
    Residues that don't occur receive a frequency of 0"""
    index = resnums.index(resnum) + 2  # plus 2 because sequence info starts at column 3.
    seq_distrib = list()
    with open(seqfile, 'r') as seqs:
        for line in seqs:
            seq_distrib.append(one_letter_code[line.split()[index]])
    total_count = len(seq_distrib)
    seq_distrib_cnt = Counter(seq_distrib)
    seq_dict = {aa: count/total_count for aa, count in seq_distrib_cnt.items()}
    for aa in aa_bkgrd_freq_biounits.keys():
        if aa not in seq_dict.keys():
            seq_dict[aa] = 0
    return seq_dict


def make_pssm_w_burial(output_dir, template_on_surface_dict, library_on_surface_dict):
    """Returns a pandas dataframe of the PSSM from results
    in TERMs output directory"""
    if output_dir[-1] != '/':
        output_dir += '/'
    frag_path = output_dir + 'fragments/'
    designscore_path = output_dir + 'designscore/'
    pssm = dict()
    beta = 10
    lambda_ = 0.31902034
    blossum = blossum62
    terms = [f.split('.')[0] for f in os.listdir(frag_path) if (f[:3] != 'hit' and f[-3:] == 'pdb')]
    for term in terms:
        try:
            rns = get_resnums(frag_path + term + '.pdb')
            is_exposed = [template_on_surface_dict[r] for r in rns]  # Takes residue numbers, not residue index
            rn = int(term.split('_')[-1][1:])
            index = rns.index(rn)
            rn_is_exposed = is_exposed[index]
            if rn_is_exposed:
                aa_bkgrd_freq_ = aa_bkgrd_freq_biounits_exposed
            else:
                aa_bkgrd_freq_ = aa_bkgrd_freq_biounits_buried
            file_base = designscore_path + 't1k_' + term
            seq_dict, good_lines = zero_order_freq_w_burial(rn, rns, file_base, is_exposed, library_on_surface_dict)
            qs = get_qs(lambda_, aa_bkgrd_freq_, blossum)
            gs = get_gs(seq_dict, aa_bkgrd_freq_, qs)
            alpha = get_alpha_w_burial(file_base + '.seq', good_lines)
            Qs = get_Qs(alpha, beta, seq_dict, gs)
            pssm[rn] = calc_column_scores(lambda_, Qs, aa_bkgrd_freq_)
        except:
            print('Exception: ', term)
            traceback.print_exc()
    pssm = pd.DataFrame(pssm)
    return pssm


def make_pssm_w_burial_alt(output_dir, template_on_surface_dict, library_on_surface_dict):
    """Returns a pandas dataframe of the PSSM from results
    in TERMs output directory"""
    if output_dir[-1] != '/':
        output_dir += '/'
    frag_path = output_dir + 'fragments/'
    designscore_path = output_dir + 'designscore/'
    pssm = dict()
    beta = 10
    lambda_ = 0.31902034
    blossum = blossum62
    terms = [f.split('.')[0] for f in os.listdir(frag_path) if (f[:3] != 'hit' and f[-3:] == 'pdb')]
    for term in terms:
        try:
            rns = get_resnums(frag_path + term + '.pdb')
            is_exposed = [template_on_surface_dict[r] for r in rns]  # Takes residue numbers, not residue index
            rn = int(term.split('_')[-1][1:])
            index = rns.index(rn)
            rn_is_exposed = is_exposed[index]
            if rn_is_exposed:
                aa_bkgrd_freq_ = aa_bkgrd_freq_biounits_exposed
            else:
                aa_bkgrd_freq_ = aa_bkgrd_freq_biounits_buried
            file_base = designscore_path + 'uniq_t1k_' + term   #uniq terms to remove redundancy
            obs_dict, expect_dict = observed_counts_w_burial(rn, rns, file_base, is_exposed,
                                                             aa_bkgrd_freq_, library_on_surface_dict)
            pssm[rn] = calc_obs_expect_scores(obs_dict, expect_dict)
        except:
            print('Exception: ', term)
            traceback.print_exc()
    pssm = pd.DataFrame(pssm)
    return pssm


def calc_obs_expect_scores(obs_dict, expect_dict):
    """Calcs the pssm scores for one column of pssm, see bottom of first
    paragraph, column 2 of pg 3396 of NucAcRes paper"""
    return {aa: np.log(obs_dict[aa] / expect_dict[aa]) for aa in obs_dict.keys()}


def get_resindices(line):
    line = line.strip()
    resi = line.split('[')[-1].split(']')[0]
    resi = resi.split('), (')
    resi = [r.split(',') for r in resi]
    resi = [[k.replace('(', '') for k in r] for r in resi]
    resi = [[k.replace(')', '') for k in r] for r in resi]
    resi = [[k.replace(']', '') for k in r] for r in resi]
    resi = [[int(k) for k in r] for r in resi]
    resi_list = list()
    for r in resi:
        if len(r) > 1:
            resi_list.extend(list(range(r[0], r[1] + 1)))
        elif len(r) == 1:
            resi_list.append(r)
    return resi_list


def calc_Nc_w_burial(seqfile, good_lines):
    """Calculates the mean number of residue types across the columns
    of the sequence alignment see bottom of pg 3395 of NucAcRes paper"""
    seqs = list()
    with open(seqfile, 'r') as seqf:
        for j, line in enumerate(seqf):
            if j in good_lines:
                seqs.append([one_letter_code[i] for i in line.split()[2:]])
    seqs = np.array(seqs)
    return np.mean([len(np.unique(seqs[:, i])) for i in range(seqs.shape[1])])


def get_alpha_w_burial(seqfile, good_lines):
    """Needed for calc of Q, see under formula 5 of NucAcRes paper"""
    return calc_Nc_w_burial(seqfile, good_lines) - 1


def zero_order_freq_w_burial(resnum, resnums, file_base, is_exposed, library_on_surface_dict):
    """Returns a dictionary of the AA freq at resnum from the TERM seqfile
    Residues that don't occur receive a frequency of 0"""
    index = resnums.index(resnum) + 2  # plus 2 because sequence info starts at column 3.
    rn_is_exposed = is_exposed[index - 2]
    seq_distrib = list()
    seqfile = file_base + '.seq'
    matchfile = file_base + '.match'
    good_lines = list()
    with open(seqfile, 'r') as seqs, open(matchfile, 'r') as matches:
        for i, (line_seq, line_match) in enumerate(zip(seqs, matches)):
            try:
            # test if the res is same burial status as rn_is_exposed
                pdb = line_match.split('/')[-1].split('.')[0]
                resindices = get_resindices(line_match)
                resindex = resindices[index - 2]
                if library_on_surface_dict[pdb][resindex] == rn_is_exposed:
                    seq_distrib.append(one_letter_code[line_seq.split()[index]])
                    good_lines.append(i)
            except:
                pass
    total_count = len(seq_distrib)
    if rn_is_exposed:
        exposure = 'exposed'
    else:
        exposure = 'buried'
    print(file_base, 'total_count:', total_count, exposure)
    seq_distrib_cnt = Counter(seq_distrib)
    seq_dict = {aa: count/total_count for aa, count in seq_distrib_cnt.items()}
    for aa in aa_bkgrd_freq_biounits.keys():
        if aa not in seq_dict.keys():
            seq_dict[aa] = 0
    return seq_dict, good_lines


def observed_counts_w_burial(resnum, resnums, file_base, is_exposed, bkgrd_dict, library_on_surface_dict):
    """Returns a dictionary of the AA freq at resnum from the TERM seqfile
    Residues that don't occur receive a frequency of 0"""
    index = resnums.index(resnum) + 2  # plus 2 because sequence info starts at column 3.
    rn_is_exposed = is_exposed[index - 2]
    seq_distrib = list()
    seqfile = file_base + '.seq'
    matchfile = file_base + '.match'
    good_lines = list()
    with open(seqfile, 'r') as seqs, open(matchfile, 'r') as matches:
        for i, (line_seq, line_match) in enumerate(zip(seqs, matches)):
            try:
            # test if the res is same burial status as rn_is_exposed
                pdb = line_match.split('/')[-1].split('.')[0]
                resindices = get_resindices(line_match)
                resindex = resindices[index - 2]
                if library_on_surface_dict[pdb][resindex] == rn_is_exposed:
                    seq_distrib.append(one_letter_code[line_seq.split()[index]])
            except:
                pass
    total_count = len(seq_distrib)
    if rn_is_exposed:
        exposure = 'exposed'
    else:
        exposure = 'buried'
    print(file_base, 'total_count:', total_count, exposure)
    seq_distrib_cnt = Counter(seq_distrib)
    obs_dict = {aa: count for aa, count in seq_distrib_cnt.items()}
    for aa in aa_bkgrd_freq_biounits.keys():
        if aa not in obs_dict.keys():
            obs_dict[aa] = 0

    expect_dict = {aa: (bkgrd_dict[aa]*total_count) for aa in bkgrd_dict.keys()}
    for aa, count in obs_dict.items():
        sqrt_count = np.sqrt(count)
        sqrt_expect = np.sqrt(expect_dict[aa])
        sigma = max(sqrt_count, sqrt_expect)
        obs_dict[aa] = count + sigma
        expect_dict[aa] = expect_dict[aa] + sigma

    return obs_dict, expect_dict


def observed_counts_w_burial_allresnums(resnums, file_base, is_exposed, library_on_surface_dict):
    """Returns a dictionary of the AA freq at resnum from the TERM seqfile
    Residues that don't occur receive a frequency of 0"""
    obs_dicts = dict()
    exp_dicts = dict()
    for resnum in resnums:
        index = resnums.index(resnum) + 2  # plus 2 because sequence info starts at column 3.
        rn_is_exposed = is_exposed[index - 2]
        if rn_is_exposed:
            bkgrd_dict = aa_bkgrd_freq_biounits_exposed_binary
        else:
            bkgrd_dict = aa_bkgrd_freq_biounits_buried_binary
        seq_distrib = list()
        seqfile = file_base + '.seq'
        matchfile = file_base + '.match'
        good_lines = list()
        with open(seqfile, 'r') as seqs, open(matchfile, 'r') as matches:
            for i, (line_seq, line_match) in enumerate(zip(seqs, matches)):
                try:
                # test if the res is same burial status as rn_is_exposed
                    pdb = line_match.split('/')[-1].split('.')[0]
                    resindices = get_resindices(line_match)
                    resindex = resindices[index - 2]
                    if library_on_surface_dict[pdb][resindex] == rn_is_exposed:
                        seq_distrib.append(one_letter_code[line_seq.split()[index]])
                except:
                    pass
        total_count = len(seq_distrib)
        if rn_is_exposed:
            exposure = 'exposed'
        else:
            exposure = 'buried'
        print(file_base, 'total_count:', total_count, exposure)
        seq_distrib_cnt = Counter(seq_distrib)
        obs_dict = {aa: count for aa, count in seq_distrib_cnt.items()}
        for aa in aa_bkgrd_freq_biounits.keys():
            if aa not in obs_dict.keys():
                obs_dict[aa] = 0

        expect_dict = {aa: (bkgrd_dict[aa]*total_count) for aa in bkgrd_dict.keys()}
        obs_dicts[resnum] = obs_dict
        exp_dicts[resnum] = expect_dict
    return obs_dicts, exp_dicts


def make_pssm_w_burial_avg_over_resnums(output_dir, template_on_surface_dict, library_on_surface_dict):
    """Returns a pandas dataframe of the PSSM from results
    in TERMs output directory"""
    if output_dir[-1] != '/':
        output_dir += '/'
    frag_path = output_dir + 'fragments/'
    designscore_path = output_dir + 'designscore/'
    pssm = dict()
    all_obs_dict = defaultdict(dict)
    all_exp_dict = defaultdict(dict)
    terms = [f.split('.')[0] for f in os.listdir(frag_path) if (f[:3] != 'hit' and f[-3:] == 'pdb')]
    for term in terms:
        try:
            rns = get_resnums(frag_path + term + '.pdb')
            is_exposed = [template_on_surface_dict[r] for r in rns]  # Takes residue numbers, not residue index
            file_base = designscore_path + 'uniq_t1k_' + term   #uniq terms to remove redundancy
            obs_dicts, exp_dicts = observed_counts_w_burial_allresnums(rns, file_base, is_exposed,
                                                              library_on_surface_dict)
            for rn in obs_dicts.keys():
                for aa in obs_dicts[rn].keys():
                    try:
                        all_obs_dict[rn][aa] += obs_dicts[rn][aa]
                    except:
                        all_obs_dict[rn][aa] = 0
                    try:
                        all_exp_dict[rn][aa] += exp_dicts[rn][aa]
                    except:
                        all_exp_dict[rn][aa] = 0

        except:
            print('Exception: ', term)
            traceback.print_exc()

    pssm = calc_obs_expect_scores_allresnums(all_obs_dict, all_exp_dict)
    pssm = pd.DataFrame(pssm)
    return pssm


def calc_obs_expect_scores_allresnums(all_obs_dict, all_exp_dict):
    """Calcs the pssm scores for one column of pssm, see bottom of first
    paragraph, column 2 of pg 3396 of NucAcRes paper"""
    pssm = dict()
    for rn in all_obs_dict.keys():
        obs_dict = all_obs_dict[rn]
        expect_dict = all_exp_dict[rn]
        for aa, count in obs_dict.items():
            sqrt_count = np.sqrt(count)
            sqrt_expect = np.sqrt(expect_dict[aa])
            sigma = max(sqrt_count, sqrt_expect)
            obs_dict[aa] = count + sigma
            expect_dict[aa] = expect_dict[aa] + sigma
        pssm[rn] = {aa: np.log(obs_dict[aa] / expect_dict[aa]) for aa in obs_dict.keys()}
    return pssm


def observed_counts_w_burial_allresnums_layers(resnums, file_base, dist_buried, library_dist_surface_dict):
    """Returns a dictionary of the AA freq at resnum from the TERM seqfile
    Residues that don't occur receive a frequency of 0"""
    obs_dicts = dict()
    exp_dicts = dict()
    for resnum in resnums:
        index = resnums.index(resnum) + 2  # plus 2 because sequence info starts at column 3.
        rn_dist = dist_buried[index - 2]
        if rn_dist <= 1:
            bkgrd_dict = aa_bkgrd_freq_biounits_exposed_ternary
            cond = 'exposed'
        elif (rn_dist > 1) and (rn_dist <= 3):
            bkgrd_dict = aa_bkgrd_freq_biounits_intermediate_ternary
            cond = 'inter'
        else:
            bkgrd_dict = aa_bkgrd_freq_biounits_buried_ternary
            cond = 'buried'
        seq_distrib = list()
        seqfile = file_base + '.seq'
        matchfile = file_base + '.match'
        good_lines = list()
        with open(seqfile, 'r') as seqs, open(matchfile, 'r') as matches:
            for i, (line_seq, line_match) in enumerate(zip(seqs, matches)):
                try:
                    pdb = line_match.split('/')[-1].split('.')[0]
                    resindices = get_resindices(line_match)
                    resindex = resindices[index - 2]
                    lib_dist = float(library_dist_surface_dict[pdb][resindex])
                    if lib_dist <= 1:
                        lib_con = 'exposed'
                    elif (lib_dist > 1) and (lib_dist <= 3):
                        lib_con = 'inter'
                    else:
                        lib_con = 'buried'
                    if lib_con == cond:
                        seq_distrib.append(one_letter_code[line_seq.split()[index]])
                except:
                    pass
        total_count = len(seq_distrib)
        exposure = cond
        print(file_base, 'total_count:', total_count, exposure)
        seq_distrib_cnt = Counter(seq_distrib)
        obs_dict = {aa: count for aa, count in seq_distrib_cnt.items()}
        for aa in aa_bkgrd_freq_biounits.keys():
            if aa not in obs_dict.keys():
                obs_dict[aa] = 0

        expect_dict = {aa: (bkgrd_dict[aa]*total_count) for aa in bkgrd_dict.keys()}
        obs_dicts[resnum] = obs_dict
        exp_dicts[resnum] = expect_dict
    return obs_dicts, exp_dicts


def make_pssm_w_burial_avg_over_resnums_layers(output_dir, template_dist_surface_dict, library_dist_surface_dict):
    """Returns a pandas dataframe of the PSSM from results
    in TERMs output directory"""
    if output_dir[-1] != '/':
        output_dir += '/'
    frag_path = output_dir + 'fragments/'
    designscore_path = output_dir + 'designscore/'
    all_obs_dict = defaultdict(dict)
    all_exp_dict = defaultdict(dict)
    terms = [f.split('.')[0] for f in os.listdir(frag_path) if (f[:3] != 'hit' and f[-3:] == 'pdb')]
    for term in terms:
        try:
            rns = get_resnums(frag_path + term + '.pdb')
            dist_buried = [template_dist_surface_dict[r] for r in rns]  # Takes residue numbers, not residue index
            file_base = designscore_path + 'uniq_t1k_' + term   #uniq terms to remove redundancy
            obs_dicts, exp_dicts = observed_counts_w_burial_allresnums_layers(rns, file_base, dist_buried,
                                                              library_dist_surface_dict)
            for rn in obs_dicts.keys():
                for aa in obs_dicts[rn].keys():
                    try:
                        all_obs_dict[rn][aa] += obs_dicts[rn][aa]
                    except:
                        all_obs_dict[rn][aa] = 0
                    try:
                        all_exp_dict[rn][aa] += exp_dicts[rn][aa]
                    except:
                        all_exp_dict[rn][aa] = 0

        except:
            print('Exception: ', term)
            traceback.print_exc()

    pssm = calc_obs_expect_scores_allresnums(all_obs_dict, all_exp_dict)
    pssm = pd.DataFrame(pssm)
    return pssm


def observed_counts_w_burial_allresnums_layers_ss_prop(resnums, file_base, dist_buried, library_dist_surface_dict, dssp_dict):
    """Returns a dictionary of the AA freq at resnum from the TERM seqfile
    Residues that don't occur receive a frequency of 0"""
    obs_dicts = dict()
    exp_dicts = dict()
    for resnum in resnums:
        dssp_code = dssp_dict[(resnum, 'A')]
        index = resnums.index(resnum) + 2  # plus 2 because sequence info starts at column 3.
        rn_dist = dist_buried[index - 2]
        if rn_dist <= 1:
            bkgrd_dict = aa_bkgrd_freq_biounits_exposed_ternary_ss
            cond = 'exposed'
        elif (rn_dist > 1) and (rn_dist <= 3):
            bkgrd_dict = aa_bkgrd_freq_biounits_intermediate_ternary_ss
            cond = 'inter'
        else:
            bkgrd_dict = aa_bkgrd_freq_biounits_buried_ternary_ss
            cond = 'buried'
        seq_distrib = list()
        seqfile = file_base + '.seq'
        matchfile = file_base + '.match'
        good_lines = list()
        with open(seqfile, 'r') as seqs, open(matchfile, 'r') as matches:
            for i, (line_seq, line_match) in enumerate(zip(seqs, matches)):
                try:
                # test if the res is same burial status as rn_is_exposed
                    pdb = line_match.split('/')[-1].split('.')[0]
                    resindices = get_resindices(line_match)
                    resindex = resindices[index - 2]
                    lib_dist = float(library_dist_surface_dict[pdb][resindex])
                    if lib_dist <= 1:
                        lib_con = 'exposed'
                    elif (lib_dist > 1) and (lib_dist <= 3):
                        lib_con = 'inter'
                    else:
                        lib_con = 'buried'
                    if lib_con == cond:
                        seq_distrib.append(one_letter_code[line_seq.split()[index]])
                except:
                    pass
        total_count = len(seq_distrib)
        exposure = cond
        print(file_base, 'total_count:', total_count, exposure)
        seq_distrib_cnt = Counter(seq_distrib)
        obs_dict = {aa: count for aa, count in seq_distrib_cnt.items()}
        for aa in aa_bkgrd_freq_biounits.keys():
            if aa not in obs_dict.keys():
                obs_dict[aa] = 0

        expect_dict = {aa: (bkgrd_dict[dssp_code][aa] * total_count)
                       for aa in bkgrd_dict[dssp_code].keys()}
        obs_dicts[resnum] = obs_dict
        exp_dicts[resnum] = expect_dict
    return obs_dicts, exp_dicts


def make_pssm_w_burial_avg_over_resnums_layers_ss_prop(output_dir, dssp_file, template_dist_surface_dict,
                                                       library_dist_surface_dict):
    """Returns a pandas dataframe of the PSSM from results
    in TERMs output directory"""
    if output_dir[-1] != '/':
        output_dir += '/'
    frag_path = output_dir + 'fragments/'
    designscore_path = output_dir + 'designscore/'
    all_obs_dict = defaultdict(dict)
    all_exp_dict = defaultdict(dict)
    terms = [f.split('.')[0] for f in os.listdir(frag_path) if (f[:3] != 'hit' and f[-3:] == 'pdb')]
    dssp_dict = parse_dssp(dssp_file)
    for term in terms:
        try:
            rns = get_resnums(frag_path + term + '.pdb')
            dist_buried = [template_dist_surface_dict[r] for r in rns]  # Takes residue numbers, not residue index
            file_base = designscore_path + 'uniq_t1k_' + term   #uniq terms to remove redundancy
            obs_dicts, exp_dicts = observed_counts_w_burial_allresnums_layers_ss_prop(rns, file_base, dist_buried,
                                                              library_dist_surface_dict, dssp_dict)
            for rn in obs_dicts.keys():
                for aa in obs_dicts[rn].keys():
                    try:
                        all_obs_dict[rn][aa] += obs_dicts[rn][aa]
                    except:
                        all_obs_dict[rn][aa] = 0
                    try:
                        all_exp_dict[rn][aa] += exp_dicts[rn][aa]
                    except:
                        all_exp_dict[rn][aa] = 0

        except:
            print('Exception: ', term)
            traceback.print_exc()

    pssm = calc_obs_expect_scores_allresnums(all_obs_dict, all_exp_dict)
    pssm = pd.DataFrame(pssm)
    return pssm



def zero_order_propensity(seq_dict):
    """Uses the database AA background frequency to determine the AA propensity"""
    return {aa: freq/aa_bkgrd_freq_biounits[aa] for aa, freq in seq_dict.items()}


def zero_order_propensity_ss_correction(seq_dict, dssp_code):
    """Uses the database AA background frequency to determine the AA propensity"""
    return {aa: freq/aa_bkgrd_freq_biounits[aa]/aa_dssp_propensities[dssp_code][aa]
            for aa, freq in seq_dict.items()}


def get_aa_bkgr_freq_db(db_directory):
    """Gets the background AA frequency from the master PDS database."""
    aa_dict = defaultdict(int)
    total_ds = len([d for d in os.listdir(db_directory) if d != 'list'])
    for i, d in enumerate([d for d in os.listdir(db_directory) if d != 'list']):
        print('directory: ' + d + ', ' + str(i) + ' of ' + str(total_ds))
        for f in os.listdir(d):
            aaset = set()
            with open(db_directory + '/' + d + '/' + f, 'rb') as infile:
                for line in infile:
                    try:
                        line = line.decode("utf-8")
                        spl = line.split()
                        if spl[0] == "ATOM":
                            aaset.add((spl[3], spl[5]))
                    except UnicodeDecodeError:
                        pass
                    except IndexError:
                        pass
            for aa, resnum in aaset:
                aa_dict[aa] += 1
    totalaa = sum(aa_dict.values())
    aa_bkgr_freq = defaultdict(int)
    for aa, c in aa_dict.items():
        aa_bkgr_freq[one_letter_code[aa]] += c / totalaa
    return aa_bkgr_freq


def get_terms_propensities(output_dir):
    """Returns all zero-order TERMs propensities of structure"""
    if output_dir[-1] != '/':
        output_dir += '/'
    frag_path = output_dir + 'fragments/'
    designscore_path = output_dir + 'designscore/'
    terms_propensities = dict()
    terms = [f.split('.')[0] for f in os.listdir(frag_path) if f[:3] != 'hit']
    for term in terms:
        rns = get_resnums(frag_path + term + '.pdb')
        rn = int(term.split('_')[-1][1:])
        seq_dict = zero_order_freq(rn, rns, designscore_path + 't1k_' + term + '.seq')
        prop_dict = zero_order_propensity(seq_dict)
        terms_propensities[rn] = prop_dict
    return terms_propensities


def make_dssp_file(pdbfile):
    pr.execDSSP(pdbfile)


def get_terms_propensities_ss_correction(output_dir, dssp_file):
    """Returns all zero-order TERMs propensities of structure,
    corrected by sec struct propensity."""
    if output_dir[-1] != '/':
        output_dir += '/'
    frag_path = output_dir + 'fragments/'
    designscore_path = output_dir + 'designscore/'
    terms_propensities = dict()
    terms = [f.split('.')[0] for f in os.listdir(frag_path)]
    dssp_dict = parse_dssp(dssp_file)
    for term in terms:
        rns = get_resnums(frag_path + term + '.pdb')
        rn = int(term.split('_')[-1][1:])
        chid = term.split('_')[-1][0]
        seq_dict = zero_order_freq(rn, rns, designscore_path + 't1k_' + term + '.seq')
        dssp_code = dssp_dict[(rn, chid)]
        prop_dict = zero_order_propensity_ss_correction(seq_dict, dssp_code)
        terms_propensities[rn] = prop_dict
    return terms_propensities


def calc_kullback_leibler(seq_dict):
    """Uses the database AA background frequency to kullback leibler divergence
     (actually it is the average self-information) for position"""
    return {aa: freq * np.log(freq / aa_bkgrd_freq_db[aa]) for aa, freq in seq_dict.items()}


def get_terms_kullback_leibler(output_dir):
    """Returns all zero-order TERMs propensities of structure"""
    if output_dir[-1] != '/':
        output_dir += '/'
    frag_path = output_dir + 'fragments/'
    designscore_path = output_dir + 'designscore/'
    terms_propensities = dict()
    terms = [f.split('.')[0] for f in os.listdir(frag_path)]
    for term in terms:
        rns = get_resnums(frag_path + term + '.pdb')
        rn = int(term.split('_')[-1][1:])
        seq_dict = zero_order_freq(rn, rns, designscore_path + 't1k_' + term + '.seq')
        si_dict = calc_kullback_leibler(seq_dict)
        terms_propensities[rn] = si_dict
    return terms_propensities

