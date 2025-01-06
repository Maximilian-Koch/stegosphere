
#The standard End-of-message signifier - gets converted to binary. Long enough for all standard use cases.
DELIMITER_MESSAGE = '###END###'


#32 bit for metadata length should be sufficient for almost all usecases
#Set higher if hidden message exceeds ~0.5GB
#Set lower if hidden message needs to be minimized
METADATA_LENGTH_IMAGE = 32
METADATA_LENGTH_AUDIO = 32
METADATA_LENGTH_VIDEO = 32
METADATA_LENGTH_TTF = 32

METADATA_LENGTH_LSB = 32
METADATA_LENGTH_VD = 32
