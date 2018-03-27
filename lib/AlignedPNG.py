PNG_HEADER = b"\x89PNG\r\n\x1a\n"

import string
import struct
import zlib
import pickle

class Chunk(object):
    def __init__(self, name=None, data=None):
        self.length = 0
        self.crc = 0
        self.name = name if name else "noNe"
        self.data = data if data else b""

    @classmethod
    def load(cls, data):
        """Load a chunk including header and footer"""
        inst = cls()
        if len(data) < 12:
            msg = "Chunk-data too small"
            raise ValueError(msg)

        # chunk header & data
        (inst.length, raw_name) = struct.unpack("!I4s", data[0:8])
        inst.data = data[8:-4]
        inst.verify_length()
        inst.name = raw_name.decode("ascii")
        inst.verify_name()

        # chunk crc
        inst.crc = struct.unpack("!I", data[8+inst.length:8+inst.length+4])[0]
        inst.verify_crc()

        return inst

    def dump(self, auto_crc=True, auto_length=True):
        """Return the chunk including header and footer"""
        if auto_length: self.update_length()
        if auto_crc: self.update_crc()
        self.verify_name()
        return struct.pack("!I", self.length) + self.get_raw_name() + self.data + struct.pack("!I", self.crc)

    def verify_length(self):
        if len(self.data) != self.length:
            msg = "Data length ({}) does not match length in chunk header ({})".format(len(self.data), self.length)
            raise ValueError(msg)
        return True

    def verify_name(self):
        for c in self.name:
            if c not in string.ascii_letters:
                msg = "Invalid character in chunk name: {}".format(repr(self.name))
                raise ValueError(msg)
            return True

    def verify_crc(self):
        calculated_crc = self.get_crc()
        if self.crc != calculated_crc:
            msg = "CRC mismatch: {:08X} (header), {:08X} (calculated)".format(self.crc, calculated_crc)
            raise ValueError(msg)
        return True

    def update_length(self):
        self.length = len(self.data)

    def update_crc(self):
        self.crc = self.get_crc()

    def get_crc(self):
        return zlib.crc32(self.get_raw_name() + self.data)

    def get_raw_name(self):
        return self.name if isinstance(self.name, bytes) else self.name.encode("ascii")

    # name helper methods

    def ancillary(self, set=None):
        """Set and get ancillary=True/critical=False bit"""
        if set is True:
            self.name[0] = self.name[0].lower()
        elif set is False:
            self.name[0] = self.name[0].upper()
        return self.name[0].islower()

    def private(self, set=None):
        """Set and get private=True/public=False bit"""
        if set is True:
            self.name[1] = self.name[1].lower()
        elif set is False:
            self.name[1] = self.name[1].upper()
        return self.name[1].islower()

    def reserved(self, set=None):
        """Set and get reserved_valid=True/invalid=False bit"""
        if set is True:
            self.name[2] = self.name[2].upper()
        elif set is False:
            self.name[2] = self.name[2].lower()
        return self.name[2].isupper()

    def safe_to_copy(self, set=None):
        """Set and get save_to_copy=True/unsafe=False bit"""
        if set is True:
            self.name[3] = self.name[3].lower()
        elif set is False:
            self.name[3] = self.name[3].upper()
        return self.name[3].islower()

    def __str__(self):
        return "<Chunk '{name}' length={length} crc={crc:08X}>".format(**self.__dict__)

class IEND(Chunk):
    def __init__(self):
        super().__init__("IEND")

    def dump(self):
        if len(self.data) != 0:
            msg = "IEND has data which is not allowed"
            raise ValueError(msg)
        if self.length != 0:
            msg = "IEND data lenght is not 0 which is not allowed"
            raise ValueError(msg)
        return super().dump()

    def __str__(self):
        return "<Chunk:IEND>".format(**self.__dict__)

class FaceswapChunk(Chunk):
    def __init__(self, dict_data=None):
        super().__init__("fcWp")
        self.dict_data = dict_data       

    def setDictData(self, dict_data):
        self.dict_data = dict_data
        
    def getDictData(self):
        return self.dict_data
        
    @classmethod
    def load(cls, data):
        inst = super().load(data)
        inst.dict_data = pickle.loads( inst.data )        
        return inst
        
    def dump(self):
        self.data = pickle.dumps (self.dict_data)
        return super().dump()
        
chunk_map = {
    b"fcWp": FaceswapChunk,
    b"IEND": IEND
}

class AlignedPNG(object):
    def __init__(self):
        self.data = b""
        self.length = 0
        self.chunks = []

    @staticmethod
    def load(data):

        try:
            with open(data, "rb") as f:
                data = f.read()
        except:
            raise FileNotFoundError(data)
    
        inst = AlignedPNG()
        inst.data = data
        inst.length = len(data)
        
        if data[0:8] != PNG_HEADER:
            msg = "No Valid PNG header"
            raise ValueError(msg)

        chunk_start = 8
        while chunk_start < inst.length:
            (chunk_length, chunk_name) = struct.unpack("!I4s", data[chunk_start:chunk_start+8])
            chunk_end = chunk_start + chunk_length + 12

            chunk = chunk_map.get(chunk_name, Chunk).load(data[chunk_start:chunk_end])
            inst.chunks.append(chunk)
            chunk_start = chunk_end

        return inst
        
        
    def save(self, filename):
        try:
            with open(filename, "wb") as f:
                f.write ( self.dump() )
        except:
            raise Exception( 'cannot save %s' % (filename) )

    def dump(self):
        data = PNG_HEADER
        for chunk in self.chunks:
            data += chunk.dump()
        return data
                
    def getFaceswapDictData(self):        
        for chunk in self.chunks:
            if type(chunk) == FaceswapChunk:
                return chunk.getDictData()
        return None
                
    def setFaceswapDictData (self, dict_data=None):
        for chunk in self.chunks:
            if type(chunk) == FaceswapChunk:
                self.chunks.remove(chunk)
                break
    
        if not dict_data is None:
            chunk = FaceswapChunk(dict_data)
            self.chunks.insert(-1, chunk)
        
    def __str__(self):
        return "<PNG length={length} chunks={}>".format(len(self.chunks), **self.__dict__)
