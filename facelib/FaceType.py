from enum import IntEnum
class FaceType(IntEnum):
    HALF = 0,
    FULL = 1,
    HEAD = 2,
    AVATAR = 3,
    QTY = 4

    @staticmethod
    def fromString (s):
        if s == 'half_face':
            return FaceType.HALF
        elif s == 'full_face':
            return FaceType.FULL
        elif s == 'head':
            return FaceType.HEAD
        elif s == 'avatar':
            return FaceType.AVATAR
        else:
            raise Exception ('FaceType.fromString value error')
            
    @staticmethod        
    def toString (face_type):
        if face_type == FaceType.HALF:
            return 'half_face'
        elif face_type == FaceType.FULL:
            return 'full_face'
        elif face_type == FaceType.HEAD:
            return 'head'
        elif face_type == FaceType.AVATAR:
            return 'avatar'
        else:
            raise Exception ('FaceType.toString value error')
