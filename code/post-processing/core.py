'''
@component: NLP_NER_SurveyPostProcessing.
@author: Sergio (u1085404).
@summary: Core post processing functions.
@project: NLP/NER Toolkit Survey.
# History Update:
#    2021-05-11: creation
#    2021-05-26: MUC metrics and config.json
#    2021-06-01: generalisation -> multiple result files and multiple data sets.
#    2021-06-28: Calculate of the "Exact Match" metrics | Reference: <http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/>
#    2021-06-28: Adding more datasets
#    2021-06-30: Adding more metrics | Reference: "MUC-5 Scoring System: Evaluation Metrics"
#    2021-07-07: MUC-5 metrics for a dataset.
#    2021-08-19: Add "GUM", "SEC_Filings", "NIST", "WNUT17", "re3d", and "feng" datasets.  *Tags.merge()* main loop correction.
#    2021-08-20: sampling() function for mappings validation.  Generation of the "merged-tags-by-category-file" (for a specific dataset).
#    2021-10-25~27: category mappings based on the latest updated model and algorithm change.
#    2021-11-22: Add columns "Group" and "TNNT-Category" to the DMC ID tuple structure.  Add *gs18* one-to-one equivalency to *tnnt*.
#    2021-12-16: General stats about the datasets (tags).
'''


# ==================================================================================================
# Core libraries (packages):
import os
import sys
import json
import random
import locale
import datetime
import itertools # for dataset stats
import operator  # for dataset stats
from pathlib import Path
from enum import Enum

locale.setlocale(locale.LC_ALL, '') # auto (default regional settings)


# ==================================================================================================
# Utils:
class Utils:
    # running time stamps:
    dt_begin, dt_end = datetime.datetime.now(), datetime.datetime.now()

    
    CONFIG_JSON_FILE = os.path.dirname(os.path.realpath(__file__)) + "/config.json"
    with open(CONFIG_JSON_FILE) as _config_json_f: # mode="rt" (default)
        _config = json.loads(_config_json_f.read())

    DATASET_ID   = str(_config["$-exec"]) # default
    DATASET      = _config["datasets"][DATASET_ID] if (DATASET_ID != "*") else {}
    MODEL_GROUPS = _config["model-groups"]
    MAPPINGS     = _config["mappings"]
    TNNT = "tnnt:"
    
    @staticmethod
    def getMappedGroup(model):
        for (group, models) in Utils.MODEL_GROUPS.items():
            if model in models:
                return group
        return None
    
    @staticmethod
    def getMappedTnntCategories(model, category) -> list:
        gs18 = "gs18:"
        def get_gs18_tnnt_equivalent(gc_pair):
            if (gc_pair.startswith(gs18)):
                for (mapped_gc, mappings) in Utils.MAPPINGS.items():
                    if mapped_gc.startswith(Utils.TNNT) and (gc_pair in mappings):
                        return mapped_gc # returns the first encountered mapping
            return None
        def lookInGroups(gc_pair, onlyInTNNT=True) -> list:
            _categories_list = [ ]
            for (mapped_gc, mappings) in Utils.MAPPINGS.items():
                if (onlyInTNNT):
                    if mapped_gc.startswith(Utils.TNNT) and (gc_pair in mappings):
                        #print(f"... {mapped_gc}")
                        _categories_list.append(mapped_gc)
                else:
                    if (gc_pair in mappings):
                        #print(f"... {mapped_gc}")
                        _categories_list.append(mapped_gc)
            #print(f"... {_categories_list}")
            return _categories_list
        def lookFromGCpairList(pair) -> list:
            gc_pair_list = Utils.MAPPINGS[pair]
            _categories_list = [ ]
            for element in gc_pair_list:
                for tnnt_category in lookInGroups(element):
                    _categories_list.append(tnnt_category)
            #print(f"... {_categories_list}")
            return _categories_list
        group = Utils.getMappedGroup(model)
        #print(f"model='{model}'; category='{category}' --> group='{group}'")
        if group is None: # group not found
            return [ ]
        gc_pair = f"{group}:{category}"
        tnnt_categories_list = [ ]
        gs18_equivalent = get_gs18_tnnt_equivalent(gc_pair)
        if (gs18_equivalent is not None): # gs18 --(equivalency)--> tnnt (one to one)
            tnnt_categories_list.append(gs18_equivalent)
            return tnnt_categories_list
        if (gc_pair in Utils.MAPPINGS):
            tnnt_categories_list = lookFromGCpairList(gc_pair)
        else:
            _categories_list = lookInGroups(gc_pair, False)
            for element in _categories_list:
                if element.startswith(Utils.TNNT):
                    tnnt_categories_list.append(element)
                else:
                    for item in lookFromGCpairList(element):
                        tnnt_categories_list.append(item)
        return tnnt_categories_list
    
    @staticmethod
    def chkMappingsFromTags(model, category, mappings) -> bool:
        set_of_mappings = set(mappings)
        #print(f"\n\nMappings from tags: {set_of_mappings}")
        set_of_TNNT_categories = set(Utils.getMappedTnntCategories(model, category))
        #print(f"TNNT categories set: {set_of_TNNT_categories}")
        found = set_of_mappings.intersection(set_of_TNNT_categories)
        #print(f"Intersection: {found} --> Found? {'True' if found else 'False'}") 
        return found
    
    @staticmethod
    def expandTnntCategoryForModel(model, tnntCategoryListFromTags) -> set: # Used for
        ''' 
        # INPUT:
        #     * model="nltk_model"
        #     * tnntCategoryListFromTags = [ "tnnt:Organisation" ]
        # OUTPUT:
        #    * set("ORGANIZATION", "FACILITY") # this is the categories set of "nltk_model" that maps to [ "tnnt:Organisation" ]
        '''
        checkedGCs = set()
        def lookForAllRelatedMappings(gc_pair, targetGroup) -> set:
            if (gc_pair in checkedGCs): # already checked -> return an empty set
                return set()
            categoriesSet = set()
            for (mapped_gc, mappings) in Utils.MAPPINGS.items():
                if (gc_pair in mappings):
                    #print(f"... {mapped_gc}")
                    for m in mappings:
                        if (m.startswith(targetGroup)):
                            categoriesSet.add(m[len(targetGroup):])
                            checkedGCs.add(m)
                    if (mapped_gc.startswith(targetGroup)):
                        categoriesSet.add(mapped_gc[len(targetGroup):])
                        checkedGCs.add(mapped_gc)
            #print(f"... {_categories_list}")
            return categoriesSet
        resultSet = set()
        group = Utils.getMappedGroup(model)
        if group is None: # group not found
            return ( )
        group = f"{group}:"
        for tnntCategory in tnntCategoryListFromTags:
            for mappedCategory in Utils.MAPPINGS[tnntCategory]: # list
                if (mappedCategory.startswith(group)):
                    resultSet.add(mappedCategory[len(group):])
                    checkedGCs.add(mappedCategory)
                for m in lookForAllRelatedMappings(mappedCategory, group):
                    resultSet.add(m)
        return resultSet

    @staticmethod
    def getListOfTnntCategories() -> list:
        return [ tnnt for tnnt, _ in Utils.MAPPINGS.items() if tnnt.startswith(Utils.TNNT) ]

    @staticmethod
    def getDS_id(dataset):
        return \
            Dataset.SEC_Filings if (dataset == "SEC_Filings") else \
            Dataset.NIST_IEER   if (dataset == "NIST_IEER")   else \
            Dataset.CONLL2003   if (dataset == "CONLL2003")   else \
            Dataset.WNUT17      if (dataset == "WNUT17")      else \
            Dataset.re3d        if (dataset == "re3d")        else \
            Dataset.feng        if (dataset == "feng")        else \
            Dataset.GUM         if (dataset == "GUM")         else \
            Dataset.BTC         if (dataset == "BTC")         else 0 # not found
    @staticmethod
    def getDSname(dataset):
        return "" + \
            "SEC_Filings" if (dataset == Dataset.SEC_Filings) else \
            "NIST_IEER"   if (dataset == Dataset.NIST_IEER)   else \
            "CONLL2003"   if (dataset == Dataset.CONLL2003)   else \
            "WNUT17"      if (dataset == Dataset.WNUT17)      else \
            "re3d"        if (dataset == Dataset.re3d)        else \
            "feng"        if (dataset == Dataset.feng)        else \
            "GUM"         if (dataset == Dataset.GUM)         else \
            "BTC"         if (dataset == Dataset.BTC)         else "" # not found

    @staticmethod
    def setDS(dataset):
        Utils.DATASET_ID = Utils.getDSname(dataset)
        Utils.DATASET    = Utils._config["datasets"][Utils.DATASET_ID]
        return Utils.DATASET
    
    @staticmethod
    def getSamplingPercent():
        return Utils._config["datasets"][Utils.DATASET_ID]["sampling-percent"]
    @staticmethod
    def getFolder(folder): # "outputs", "tags"
        return Utils._config["base"] + Utils._config[folder] + Utils.DATASET[folder]
    @staticmethod
    def getResultFiles() -> list:
        return Utils.DATASET["results"]
    @staticmethod
    def getCSVtupleStructure():
        return "Entity,COR,INC,PAR,SPU,MIS,IGN"
    @staticmethod
    def getDMCstructure():
        return "Dataset,Group,Model,Category,TNNT-Category"
    @staticmethod
    def getResultFileHeading():
        return  f"{Utils.getDMCstructure()}," +\
                "Possible,Actual,Wrong,Total," +\
                "Error,Precision,Recall," +\
                "Fm__PR,Fm_2PR,Fm_P2R," +\
                "Undergeneration,Overgeneration,Substitution\n"

    @staticmethod
    def printStartTimeStamp(): # Start time stamp
        print(f">> Start time: [{Utils.dt_begin}]")

    @staticmethod
    def printEndTimeStamp(delta): # End time stamp
        print(f"\n>> End time: [{Utils.dt_end}]")
        print(f">> Execution time: {delta:,.2f} seconds.")


# ==================================================================================================
# Dataset enumeration:
class Dataset(Enum):
    CONLL2003 = 1
    GUM = 2
    BTC = 3
    SEC_Filings = 4
    NIST_IEER = 5
    WNUT17 = 6
    re3d = 7
    feng = 8


# ==================================================================================================
# Metrics enumeration: concepts based on the *MUC-5 Evaluation Metrics* 
class Metric(Enum):
    COR = 1 # CORRECT           : (results.entity == tags.entity) and (results.category     IN mappings[tags.tag])
    INC = 2 # INCORRECT         : (results.entity == tags.entity) and (results.category NOT IN mappings[tags.tag])
    PAR = 3 # PARTIALLY CORRECT : (results.entity IN tags.entity) and (results.category     IN mappings[tags.tag])
    SPU = 4 # SPURIOUS          : (results.entity NOT IN tags.entity) and (results.category     IN mappings[tags.tag])
    MIS = 5 # MISSING           : <list of not matched tags>
    IGN = 6 # IGNORE            : (results.entity NOT IN tags.entity) and (results.category NOT IN mappings[tags.tag]) 


# ==================================================================================================
# Manages the output of a specific table (CSV file), per:
# (1) Dataset.
# (2) Model.
# (3) Category.
class DMC:
    __slots__ = ('id', 'dataset', 'model', 'category', 'table',
                 'totalCOR', 'totalINC', 'totalPAR', 'totalSPU', 'totalMIS', 'totalIGN', 
                 '_possible', '_actual', '_wrong', '_total')
    
    def __init__(self, dataset, model, category):
        self.dataset = dataset
        self.model = model
        self.category = category
        self.setID()
        self.table = set() # managed as a set: unordered unique objects
        self.totalCOR = 0
        self.totalINC = 0
        self.totalPAR = 0
        self.totalSPU = 0
        self.totalMIS = 0
        self.totalIGN = 0
        self._possible = 0 # POSSIBLE(POS) = COR+INC+PAR+MIS = TP+FN
        self._actual   = 0 # ACTUAL(ACT)   = COR+INC+PAR+SPU = TP+FP
        self._wrong    = 0 # WRONG(WRG)    = INC+PAR/2+MIS+SPU
        self._total    = 0 # TOTAL(TOT)    = COR+INC+PAR+MIS+SPU
    
    def setID(self):
        self.id = f"{Utils.getDSname(self.dataset)}-{self.model}-{self.category}.csv"
    
    def hasSameID(self, dataset, model, category):
        return ( (self.dataset == dataset) and (self.model == model) and (self.category == category) )
    
    def getIDtuple(self):
        # add columns: "Group"; "TNNT Category"
        group = Utils.getMappedGroup(self.model)
        tnnt_categories_set = set(Utils.getMappedTnntCategories(self.model, self.category)) # list -> set
        tnnt_category = "tnnt:"
        if (len(tnnt_categories_set) == 1):
            tnnt_category = tnnt_categories_set.pop()
        else:
            for _, c in enumerate(tnnt_categories_set):
                tnnt_category += c[5] # tnnt:_
        return f'"{Utils.getDSname(self.dataset)}","{group}","{self.model}","{self.category}","{tnnt_category}"'

    def add(self, entity, COR=0, INC=0, PAR=0, SPU=0, MIS=0, IGN=0):
        self.table.add( (entity, COR, INC, PAR, SPU, MIS, IGN) ) # add a new tuple to the set
        self.totalCOR += 1 if (COR == 1) else 0
        self.totalINC += 1 if (INC == 1) else 0
        self.totalPAR += 1 if (PAR == 1) else 0
        self.totalSPU += 1 if (SPU == 1) else 0
        self.totalMIS += 1 if (MIS == 1) else 0
        self.totalIGN += 1 if (IGN == 1) else 0
    
    def getTotalCORscore(self):
        return self.totalCOR
    def getTotalPARscore(self):
        return self.totalPAR
    def getTotalINCscore(self):
        return self.totalINC
    def getTotalMISscore(self):
        return self.totalMIS
    def getTotalSPUscore(self):
        return self.totalSPU
    def getPossibleScore(self):
        return self._possible
    def getActualScore(self):
        return self._actual
    def getWrongScore(self):
        return self._wrong
    def getTotalScore(self):
        return self._total

    def getNumOfRows(self):
        return len(self.table)

    def csv(self, folder, factTableFile):
        print(f"-> {self.id} ... ", end="")
        i = 1
        with open(folder + self.id, 'w') as outfile:
            outfile.write(f"{Utils.getCSVtupleStructure()}\n")
            for r in self.table:
                entity = r[0].replace('"', '""') # any double quotes should be escaped with two double quotes (""). <https://www.w3.org/TR/tabular-data-model/#lines>
                _tuple = f'"{entity}",{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]}\n'
                outfile.write(_tuple) # generates the DMC CSV
                factTableFile.write(f"{self.getIDtuple()},{_tuple}") # generates the dataset's fact table
                i += 1
            outfile.write(f"*TOTAL*,{self.totalCOR},{self.totalINC},{self.totalPAR},{self.totalSPU},{self.totalMIS},{self.totalIGN}")
        self._possible = self.totalCOR + self.totalINC + self.totalPAR + self.totalMIS
        self._actual   = self.totalCOR + self.totalINC + self.totalPAR + self.totalSPU
        self._wrong    = self.totalINC + (self.totalPAR/2) + self.totalMIS + self.totalSPU
        self._total    = self.totalCOR + self.totalINC + self.totalPAR + self.totalMIS + self.totalSPU
        print(f"Done. ({i+1} lines)")


# ==================================================================================================
# Manages the set of outputs.
class OutputSet:
    __slots__ = ('dataset', 'folder', 'DMC_List', 'tuples')
    
    def __init__(self, dataset):
        Utils.setDS(dataset) # sets the usage for the current dataset
        self.dataset = dataset
        self.folder = Utils.getFolder("outputs") + Utils._config["CSVs"] # for the current dataset: "../_post-processing/"
        self.DMC_List = []
        self.tuples = 0

    def newOutput(self, model, category):
        o = DMC(self.dataset, model, category)
        self.DMC_List.append(o)
        return o
    
    def getOutput(self, model, category):
        x = None
        for o in self.DMC_List:
            if o.hasSameID(self.dataset, model, category):
                x = o
                break
        if (x is None): # new output
            x = self.newOutput(model, category)
        return x
    
    def getNumberOfOutputs(self) -> int:
        return len(self.DMC_List)
    
    def getNumberOfAddedTuples(self) -> int:
        return self.tuples

    def addTuple(self, model, category, entity, metric):
        o = self.getOutput(model, category)
        COR = int(metric == Metric.COR)
        INC = int(metric == Metric.INC)
        PAR = int(metric == Metric.PAR)
        SPU = int(metric == Metric.SPU)
        MIS = int(metric == Metric.MIS)
        IGN = int(metric == Metric.IGN)
        o.add(entity, COR, INC, PAR, SPU, MIS, IGN)
        self.tuples += 1

    # We assume that all metrics have been computed, including the last one: MIS
    def generateCSVs(self):
        def F(B, P, R): # F-measure computation
            B_sqr = B**2
            num = ((B_sqr + 1.0) * P * R)
            den = ((B_sqr * P) + R)
            return (num / den) if (den > 0) else -1.0
        factTableFile = Utils.getDSname(self.dataset) + "-RESULTS" + Utils._config["fact-table-file-suffix"]
        metricsFTFile = Utils.getDSname(self.dataset) + "-RESULTS-MUC5-Metrics" + Utils._config["fact-table-file-suffix"]
        with open(self.folder + factTableFile, 'w') as ofFT_entities:
            ofFT_entities.write(f"{Utils.getDMCstructure()},{Utils.getCSVtupleStructure()}\n")
            with open(self.folder + metricsFTFile, 'w') as ofFT_metrics:
                ofFT_metrics.write(Utils.getResultFileHeading())
                for o in self.DMC_List: # for each output set
                    o.csv(self.folder, ofFT_entities) # generate the CSVs
                    # Calculate the MUC-5 metrics
                    COR = o.getTotalCORscore() # Correct
                    PAR = o.getTotalPARscore() # Partially correct
                    INC = o.getTotalINCscore() # Incorrect
                    MIS = o.getTotalMISscore() # Missing: keys (tags) that were not matched.
                    SPU = o.getTotalSPUscore() # Spurious: results that were not matched with key (tags).
                    POS = o.getPossibleScore() # ** Possible
                    ACT = o.getActualScore()   # ** Actual
                    WRG = o.getWrongScore()    # ** Wrong
                    TOT = o.getTotalScore()    # ** Total
                    MAT = (COR + (PAR * 0.5))  # Matches: correct and partial
                    CPI = (COR + PAR + INC)    # (correct + partial + incorrect)
                    # -1 indicates #DIV/0! error
                    # Primary Metrics:
                    ERR = (WRG / TOT) if (TOT > 0) else -1 # Error: % of wrong answers.
                    PRE = (MAT / ACT) if (ACT > 0) else -1 # Precision: % of actual answers given which were correct.
                    REC = (MAT / POS) if (POS > 0) else -1 # Recall: % of possible answers which were correct.
                    # Secondary Metrics:
                    UND = (MIS / POS) if (POS > 0) else -1 # Undergeneration
                    OVG = (SPU / ACT) if (ACT > 0) else -1 # Overgeneration
                    SUB = (MAT / CPI) if (CPI > 0) else -1 # Substitution
                    # F-measures:
                    Fm__PR = F(1.0, PRE, REC) # For recall and precision are equally important.
                    Fm_2PR = F(0.5, PRE, REC) # For recall half  as important as precision.
                    Fm_P2R = F(2.0, PRE, REC) # For recall twice as important as precision.
                    ofFT_metrics.write(\
                        f"{o.getIDtuple()}," +\
                        f"{POS},{ACT},{WRG:.2f},{TOT}," +\
                        f"{ERR:.6f},{PRE:.6f},{REC:.6f}," +\
                        f"{Fm__PR:.6f},{Fm_2PR:.6f},{Fm_P2R:.6f}," +\
                        f"{UND:.6f},{OVG:.6f},{SUB:.6f}\n")
            print(f"-> {metricsFTFile} ... Done.")
        print(f"-> {factTableFile} ... Done.")


# ==================================================================================================
# Manages the tags as part of the post-processing input.

class Tags:
    FILE_MAPPINGS    = Utils._config["mappings-file"]
    FILE_MERGED_TAGS = Utils._config["merged-tags-file"]
    FILE_MERGED_TAGS_BY_CATEGORY = Utils._config["merged-tags-by-category-file"]
    __slots__ = ('dataset', 'folder', 'files', 'tags', 'mergedTags', 'mergedTagsByCategory', 'mappings', 'matchedTags', 'models')

    def __init__(self, folder, dataset):
        self.folder  = folder
        self.dataset = dataset
        self.tags        = [] # list of tags obtain from the files.
        self.mergedTags  = {} # dict of merged tags from the original compiled list.
        self.mergedTagsByCategory  = {} # dict of merged tags from the original compiled list by category (for validation mappings purposes).
        self.mappings    = {} # dict of the mappings: tagged categories with the TNNT categories.
        self.matchedTags = {} # dict of the matched tags per model.
        self.models      = {} # dict of the processed models.  For each model, we keep a set of the analysed categories.
        mappings_file  , self.mappings   = self.loadMappings()
        mergedTags_file, self.mergedTags = self.loadMergedTags()
        mergedTagsByCategory_file, self.mergedTagsByCategory = self.loadMergedTagsByCategory()
        self.files = [ f for f in Path(self.folder).glob("*.json") \
                            if (f.is_file() and \
                                (str(f) != str(mappings_file)) and \
                                (str(f) != str(mergedTags_file)) and \
                                (str(f) != str(mergedTagsByCategory_file)) ) ] # all the tag files excluding the mappings, mergedTags, and mergedTagsByCategory files
        self.loadTags()
        if not(self.mappings): # the mappings are empty
            self.createInitialMappings()
        print("")

    def loadFromFile(self, struct, filename, desc):
        _file = Path(self.folder + filename)
        struct = {}
        if _file.exists():
            print(f"Loading *{desc}* from [{str(_file)}]... ", end="")
            with _file.open() as f: # mode="rt" (default)
                struct = json.loads(f.read())
            print(f"{len(struct):n} {desc} were loaded.")
        else:
            print(f"The *{desc}* file was not found.")
        return _file, struct
    def loadMappings(self):
        return self.loadFromFile(self.mappings,   Tags.FILE_MAPPINGS,    "mappings")
    def loadMergedTags(self):
        return self.loadFromFile(self.mergedTags, Tags.FILE_MERGED_TAGS, "merged tags")
    def loadMergedTagsByCategory(self):
        return self.loadFromFile(self.mergedTagsByCategory, Tags.FILE_MERGED_TAGS_BY_CATEGORY, "merged tags by category")
    
    def createInitialMappings(self):
        mappings_file = Path(self.folder + Tags.FILE_MAPPINGS)
        print(f"Generating initial mappings file ({str(mappings_file)})...")
        for pair in self.tags:
            self.mappings[ str(pair[1])[2:] ] = [] # array of categories to be edited: "B-PER" --> "PER"
        with open(str(mappings_file), 'w') as outfile:
            outfile.write(json.dumps(self.mappings, indent=4))
        print(f"{self.howManyMappings():n} mapping elements were created.")

    def loadTags(self):
        print(f"Loading the *tags* from {self.folder}...")
        self.tags = []
        for file in self.files: # from the list of files
            print(f"\t+ Adding {str(file)}...")
            with file.open() as f: # mode="rt" (default)
                tagFile = json.loads(f.read())
            for pair in tagFile:
                self.tags.append(pair)
        print(f"All *tags* have been loaded.  Total number of *tags*: {self.howManyTags():n}")
        if not(self.mergedTags): # no merged tags were found
            self.merge()
        if not(self.mergedTagsByCategory): # no merged tags by category were found
            self.mergeByCategory()

    def printTags(self):
        print(f"Loaded *Tags*:\n{json.dumps(self.tags, indent=4)}\n")
    def printMappings(self):
        print(f"Loaded *Mappings*:\n{json.dumps(self.mappings, indent=4)}\n")
    def printMergedTags(self):
        print(f"Loaded *Merged Tags*:\n{json.dumps(self.mergedTags, indent=4)}\n")
    def printMergedTagsByCategory(self):
        print(f"Loaded *Merged Tags By Category*:\n{json.dumps(self.mergedTagsByCategory, indent=4)}\n")
    def printMatchedTags(self):
        print(f"Loaded *Matched Tags*:\n{json.dumps(self.matchedTags, indent=4)}\n")

    def getTags(self):
        return self.tags
    def getMergedTags(self):
        return self.mergedTags
    def getMergedTagsByCategory(self):
        return self.mergedTagsByCategory
    def getMappings(self):
        return self.mappings

    def howManyTags(self) -> int:
        return len(self.tags)
    def howManyMergedTags(self) -> int:
        return len(self.mergedTags)
    def howManyMatchedTagsIn(self, model) -> int:
        return len(self.matchedTags[model])
    def howManyMappings(self) -> int:
        return len(self.mappings)

    def merge(self):
        def inrange(i) -> bool:
            return (i < self.howManyTags())
        def Next(i):
            return self.tags[i] if inrange(i) else []
        def add(entity, tag) -> None:
            # print(f"\t+ [tag #{i}]: {{{entity.encode('ascii', 'ignore')} : {tag}}}")
            if (entity not in self.mergedTags): 
                self.mergedTags[entity] = [tag] # a list of tags
            else:
                (self.mergedTags[entity]).append(tag) # append the tag in the list
        print(f"Merging the *tags*...")
        self.mergedTags = {}
        i = 0
        while inrange(i): # index the list's range
            p1, p2 = Next(i), Next(i + 1)
            # For all the datasets, we have the same expected format of the tag files: { if (self.dataset == Dataset.CONLL2003): }
            entity = p1[0] # default value: from the first pair
            tag = p1[1][2:] # "B-PER" --> "PER"
            if p1 and p2: # we have the two pairs
                if ( (p1[1].startswith("B-")) and (p2[1].startswith("I-")) and (tag == p2[1][2:]) ): # ("B-LOC", "I-LOC")
                    entity = p1[0] + " " + p2[0] # merge p1 and p2
                    j = 2
                    p = Next(i + j)
                    while p: # loop the following pairs --> "I-LOC", "I-LOC"...
                        if ( (p[1].startswith("I-")) and (tag == p[1][2:]) ):
                            entity = entity + " " + p[0] # merge pj...
                            j += 1
                            p = Next(i + j)
                        else:
                            break # exit loop
                    i += j # shift by *j*
                else: # two separate entities: just add the first one
                    i += 1 # shift by 1
            add(entity, tag) # adds either p1 or (merged pairs)
            if (i == self.howManyTags()-1) and not p2: # (i is the previous to the last position) and (p2 is empty)
                break # exit main loop
        for (k,v) in self.mergedTags.items():
            self.mergedTags[k] = list(set(v)) # remove duplicate tags
        mergedTags_file = Path(self.folder + Tags.FILE_MERGED_TAGS)
        print(f"Generating merged tags file ({str(mergedTags_file)})...")
        with open(str(mergedTags_file), 'w') as outfile:
            outfile.write(json.dumps(self.mergedTags, indent=4))
        print(f"All *tags* have been merged.  {self.howManyMergedTags():n} elements were created.\nGenerated file: {str(mergedTags_file)}")
    
    def mergeByCategory(self):
        for entity, categories in self.mergedTags.items():
            for category in categories:
                if (category not in self.mergedTagsByCategory): 
                    self.mergedTagsByCategory[category] = [entity] # a list of entities
                else:
                    (self.mergedTagsByCategory[category]).append(entity) # append the entity in the list
        mergedTagsByCategory_file = Path(self.folder + Tags.FILE_MERGED_TAGS_BY_CATEGORY)
        print(f"Generating *merged tags by category* file ({str(mergedTagsByCategory_file)})...")
        with open(str(mergedTagsByCategory_file), 'w') as outfile:
            outfile.write(json.dumps(self.mergedTagsByCategory, indent=4))
        print(f"{self.howManyMergedTags():n} entities were grouped by category.\nGenerated file: {str(mergedTagsByCategory_file)}")

    def getNotMatchedTags(self, model) -> dict:
        return {
            k : self.mergedTags[k]
            for k in set(self.mergedTags) - set(self.matchedTags[model]) # set difference between the keys
        }


    def searchEntity(self, entity, category, model):
        partialMatchings = set()

        def isCategoryInMappings() -> bool:
            for listOfMappedCategories in self.mappings.values():
                if Utils.chkMappingsFromTags(model, category, listOfMappedCategories):
                    return True
            return False
        
        def doesTagMaps2Category(listOfPossibleTags) -> bool:
            for tag in listOfPossibleTags: # list/set of tags
                if Utils.chkMappingsFromTags(model, category, self.mappings[tag]):
                    return True
            return False
        
        # keeps track of the matched/used tags per model
        def addMatchedTag(tagged_entity, tagged_categories) -> None:
            if (model not in self.matchedTags):
                self.matchedTags[model] = {}
            self.matchedTags[model].update( { tagged_entity : tagged_categories } )
        
        # keeps track of the analysed models and their categories
        def addAnalysedModelCategoryPair(model, category) -> None:
            if (model not in self.models):
                self.models[model] = set()   # first time
            self.models[model].add(category) # add the category in the set (no duplicates)

        def locatePartialMatchings() -> None:
            for (tagged_entity, tagged_categories) in self.mergedTags.items():
                if  ( (len(tagged_entity) < len(       entity)) and (tagged_entity in        entity) ) or \
                    ( (len(       entity) < len(tagged_entity)) and (       entity in tagged_entity) ): # partial match
                    addMatchedTag(tagged_entity, tagged_categories) # keeps track of the matched/used tags per model
                    partialMatchings.update( set(tagged_categories) ) # the set of tagged categories (no duplicates)
        
        addAnalysedModelCategoryPair(model, category)
        if (entity in self.mergedTags):
            if doesTagMaps2Category(self.mergedTags[entity]):
                addMatchedTag(entity, self.mergedTags[entity]) # keeps track of the matched/used tags per model
                return Metric.COR
            else:
                return Metric.INC # the entity was found in the tags but the category was not found in the mappings.
        else: # the entity was not found in the tags --> (PAR, SPU, INC, IGN)
            locatePartialMatchings()
            tagMapped2Category = doesTagMaps2Category(partialMatchings)
            _isCategoryInMappings = isCategoryInMappings()
            if (tagMapped2Category): # the category was found in one of the possible tags from the partial matchings
                return Metric.PAR
            if (_isCategoryInMappings):
                return Metric.SPU
            # Revision (2021-06-10): If the category is not in the mappings (for all cases) --> IGN
            #if (partialMatchings) and not(_isCategoryInMappings):
            #    return Metric.INC --> yields some _actually_ incorrect evaluations
            #if not(partialMatchings) and not(_isCategoryInMappings):
            if not(_isCategoryInMappings):
                return Metric.IGN
        return Metric.MIS # possibly, the code will never reach this point


    def addMissingTags(self, model, _os):
        n = 0
        for (tagged_entity, tagged_categories) in self.getNotMatchedTags(model).items():
            for tag in tagged_categories:
                # for each category from the set of model categories found in the set of mapped categories:
                for category in ( self.models[model].intersection( Utils.expandTnntCategoryForModel(model, self.mappings[tag]) ) ):
                    _os.addTuple(model=model, category=category, entity=tagged_entity, metric=Metric.MIS)
                    n += 1
        return n


# ==================================================================================================
# General functions:
_sample = {}


def generateSampleFile():
    print(f"\n************************************************************************************************************************")
    suffix_sampleFile = "SAMPLE-" + Utils._config["merged-tags-by-category-file"]
    print(f'Generating *SAMPLE* file of tags: "{suffix_sampleFile}"...')
    sf = Utils._config["base"] + Utils._config["outputs"] + suffix_sampleFile
    with open(sf, 'w') as of:
        of.write(json.dumps(_sample, indent=4))
    print(f"Done.  The sample file contains {len(_sample):n} datasets.\n")


# Merge all *metricsFTFiles* into one single file that will hold the overall results
def generateSingleMergedMetricsFile():
    print(f"************************************************************************************************************************")
    suffix_metricsFiles = "-RESULTS-MUC5-Metrics" + Utils._config["fact-table-file-suffix"]
    print(f'Generating "{suffix_metricsFiles[1:]}"...')
    rf = Utils._config["base"] + Utils._config["outputs"] + suffix_metricsFiles[1:]
    tl = 1
    with open(rf, 'w') as orf:
        orf.write(Utils.getResultFileHeading())
        for dataset_name, _ in Utils._config["datasets"].items(): # process all datasets
            print(f"-> Adding {dataset_name}... ", end="")
            Utils.setDS( Utils.getDS_id(dataset_name) )
            metricsFile = Utils.getFolder("outputs") + Utils._config["CSVs"] + dataset_name + suffix_metricsFiles # for each *metricsFile*
            n = 0
            with open(metricsFile) as ofMetrics: # mode="rt" (default)
                line = ofMetrics.readline() # first line --> header
                while (line): # if (line == ""): --> EOF
                    line = ofMetrics.readline()
                    n += 1
                    orf.write(line)
            print(f"{n} lines were added.")
            tl += n
    print(f"Done.  {tl} lines were added in the CSV file.\n")


def printDS(dataset):
    print(f"************************************************************************************************************************")
    print(f'*** PROCESSING DATASET "{Utils.getDSname(dataset)}" ***\n')


def process(dataset):
    _os = OutputSet(dataset=dataset) # internally, sets the usage for the current data set
    dir_results  = Utils.getFolder("outputs")
    dir_tags     = Utils.getFolder("tags")
    file_results = Utils.getResultFiles() # list
    tags = Tags(folder=dir_tags, dataset=dataset)
    tags.printMappings()
    setOfModels = set() # initialise with an empty set.
    for file_result in file_results:
        # each result file holds a different set of models.
        print(f"\n*** Processing result file [{file_result['file']}] ***\n")
        rslts = {}
        with open( dir_results + file_result["file"] ) as _f: # mode="rt" (default)
            rslts = json.loads(_f.read())
        model, category, entity = "", "", ""
        metric = Metric.MIS # default value
        if (file_result["ignore-models"]):
            print(f"The following models will be ignored: {file_result['ignore-models']}")
        print(f"Processing entities... ", end="")
        for (entity, where) in rslts["NLP-NER-Summary"]["doc-0"].items():
            for spot in where:
                if ( ("model" in spot) and ("category" in spot) and (spot["model"] not in file_result["ignore-models"] ) ):
                    model    = spot["model"]
                    category = spot["category"]
                    setOfModels.add(model)
                    # calculate the metrics: comparing each entity with the tags:
                    metric = tags.searchEntity(entity=entity, category=category, model=model)
                    _os.addTuple(model=model, category=category, entity=entity, metric=metric)
        print(f"Done.  {_os.getNumberOfAddedTuples():n} tuples were added in a total of {_os.getNumberOfOutputs():n} CSV files\n")
        print(f"List of tags that were NOT MATCHED for each model:\n")
        for m in setOfModels:
            print(f"MODEL: {m}")
            print(f"\tNumber of matched tags _______: {tags.howManyMatchedTagsIn(m):n}")
            print(f"\tNumber of NOT MATCHED tags ___: {len(tags.getNotMatchedTags(m)):n}")
            print(f"\tProcessing MISSING entities __: {tags.addMissingTags(m, _os):n}\n")
    print(f"\nGenerating CSVs:")
    _os.generateCSVs()
    print(f"\nTOTAL: {_os.getNumberOfAddedTuples():n} tuples were added in a total of {_os.getNumberOfOutputs():n} CSV files\n\n")


def getDSinfo(dataset):
    Utils.setDS(dataset) # sets the usage for the current dataset
    dir_tags = Utils.getFolder("tags")
    tags = Tags(folder=dir_tags, dataset=dataset)
    #tags.printMergedTagsByCategory()
    mergedTagsByCategory = tags.getMergedTagsByCategory()
    return dir_tags, tags, mergedTagsByCategory

def sampling(dataset):
    _, _, mergedTagsByCategory = getDSinfo(dataset)
    samplingRate = Utils.getSamplingPercent()
    _sample[Utils.getDSname(dataset)] = {}
    print(f"Sampling rate for the dataset: {samplingRate:.2%}\nDataset sampling per category:")
    # For each category in the dataset tags (all tags; merged in one file):
    for category, entities in mergedTagsByCategory.items():
        print(f'\t* "{category}": ', end="")
        _sample[Utils.getDSname(dataset)][category] = []
        l = len(entities) # count the number of entities for the category
        s = round(l*samplingRate) if l>=51 else l-1
        print(f"{l:n} entities;\tsample ~ {s:n}.")
        for i in random.sample(range(1, l), s): # builds the sample for the category.
            _sample[Utils.getDSname(dataset)][category].append(entities[i])


all_dataset_stats = {}
def stats(dataset):
    _, tags, mergedTagsByCategory = getDSinfo(dataset)
    # Dataset metrics:
    # 1) No. of categories (their own tags) per dataset.
    # 2) No. of entities per categories (not TNNT) --> total number of entities
    # 3) Mappings of the TNNT categories: how many --> to the datasets tags | generate a number between 0..1 (%)
    # tnnt:Person: 0.333
    # tnnt:Law: 0.0021
    # ...
    # tnnt:Ordinal: 0
    # ==> arrays: [count, percentage]
    listOfTnntCategories = Utils.getListOfTnntCategories()
    tnnt_dic_init = { tnnt: [0, 0] for tnnt in listOfTnntCategories }
    stats = {
        "categories": {
            "count": tags.howManyMappings(),
            "entities-per-category": { tag: [len(entities), 0] for tag, entities in mergedTagsByCategory.items() } # count
        },
        "tnnt-mappings": tnnt_dic_init
    }
    stats["count:entities"] = sum( x[0] for x in stats["categories"]["entities-per-category"].values() ) # ["count"]
    # Calculate the percentages:
    for tag, entities in stats["categories"]["entities-per-category"].items():
        stats["categories"]["entities-per-category"][tag][1] = float(f'{entities[0]/stats["count:entities"]:.6f}') 
    sum_mappings = 0
    for tnnt in listOfTnntCategories:
        for _, mappings in tags.getMappings().items():
            if (tnnt in mappings):
                stats["tnnt-mappings"][tnnt][0] += 1 # count
                sum_mappings += 1
    for tnnt, x in stats["tnnt-mappings"].items():
        stats["tnnt-mappings"][tnnt][1] = float(f'{x[0]/sum_mappings:.6f}') # percentage
    all_dataset_stats[Utils.getDSname(dataset)] = stats # adds the dataset stats to the complete dictionary with the results
    # print(f"Stats:\n{json.dumps(all_dataset_stats, indent=4)}\n")


# ==================================================================================================
def main(function_name="process"): # process || sampling
    print(f">> Function: {function_name}\n>> Dataset to process: {'(all)' if (Utils.DATASET_ID == '*') else Utils.DATASET_ID}\n")
    if (Utils.DATASET_ID == "*"): # all datasets
        for dataset, _ in Utils._config["datasets"].items(): # all the datasets found in the configuration file
            dataset_ID = Utils.getDS_id(dataset) 
            printDS(dataset_ID)
            getattr(sys.modules[__name__], function_name)(dataset_ID)
    else: # dataset specified in "$-exec".
        getattr(sys.modules[__name__], function_name)(Utils.getDS_id(Utils.DATASET_ID))
    if (function_name == "process"):
        generateSingleMergedMetricsFile()
    if (function_name == "sampling"):
        generateSampleFile()
    print(f"************************************************************************************************************************")
    if (function_name == "stats"):
        print(f"All datasets stats:\n{json.dumps(all_dataset_stats, indent=4)}\n")


# ==================================================================================================
# main
if (__name__ == "__main__"):
    Utils.dt_begin = datetime.datetime.now()
    Utils.printStartTimeStamp()
    
    main("stats")
    #main("process")
    #print( Utils.chkMappingsFromTags("flair_fast_model", "MISC", [ "tnnt:NORP", "tnnt:GPE" ]) )
    #print( Utils.expandTnntCategoryForModel("flair_fast_model", [ "tnnt:NORP", "tnnt:GPE" ]) )

    Utils.dt_end = datetime.datetime.now()
    delta = (Utils.dt_end - Utils.dt_begin).total_seconds()
    Utils.printEndTimeStamp(delta)

# ==================================================================================================