from classes.puzzlebot.puzzlebot     import PuzzleBot, PuzzleBotModel
from classes.puzzlebot.puzzlebot_arm import PuzzleBotArm, PuzzleBotArmModel

##=========================================================================
## Inicializar Husky
##=========================================================================



##=========================================================================
## Inicializar Anymal
##=========================================================================



##=========================================================================
## Inicializar PuzzleBots
##=========================================================================

puzzlebotmodel = PuzzleBotModel()
puzzlebotarmmodel = PuzzleBotArmModel()

puzzlebotarm1 = PuzzleBotArm(puzzlebotarmmodel)
puzzlebotarm2 = PuzzleBotArm(puzzlebotarmmodel)
puzzlebotarm3 = PuzzleBotArm(puzzlebotarmmodel)

puzzlebot1 = PuzzleBot(puzzlebotmodel,puzzlebotarm1)
puzzlebot2 = PuzzleBot(puzzlebotmodel,puzzlebotarm1)
puzzlebot3 = PuzzleBot(puzzlebotmodel,puzzlebotarm1)

##=========================================================================
## Simulacion
##=========================================================================

