
class Converter():
    def patch_all( self, original, faces_detected ):
        for idx, face in faces_detected:
            original = self.patch_one_face(original, face)
        return original

    def patch_one_face( self, original, face_detected ):
        raise NotImplementedError()
