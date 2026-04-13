from enum import Enum, auto

##==========================================================

class State(Enum):
    STARTING     = auto()
    CLEARING     = auto()
    TRANSPORTING = auto()
    STACKING     = auto()
    DONE         = auto()
    ERROR        = auto()

class Coordinator:
    
    def __init__(self, husky, anymal, puzzlebots):
        self.state      = State.STARTING
        self.husky      = husky
        self.anymal     = anymal
        self.puzzlebots = puzzlebots
