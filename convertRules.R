#convertRules.R
library("BoolNet")
convertRules <- function(SBML, savepath){
  #Takes the path 'SBML' to an SBML file and converts the Boolean rules
  #to .txt in BoolNet format for further processing at the provided 'savepath' location.
  network <- loadSBML(SBML)
  saveNetwork(network, savepath)
}

convertRules(SBML="./networks/Cortical Area Development (SBML).sbml",
             savepath="./networks/rulefile_Giacomantonio.txt")

convertRules(SBML="./networks/Mammalian Cell Cycle 2006 (SBML).sbml",
             savepath="./networks/rulefile_Faure.txt")
