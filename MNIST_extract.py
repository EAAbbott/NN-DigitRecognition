
import struct
import numpy as np


def mnist_read(filename):
    """
    Extract data from MNIST idx files.
    Output is an array of N dimensions described by file header.
    File structure read according to http://yann.lecun.com/exdb/mnist/
    Data is all Big-Endian.
    """
    
    with open(filename, 'rb') as f:
        print("File to read:", filename)
        # first 4 bytes (magic number) (b'\x00\x00\x08\x03')
        #   - first 2 always 0 (H - short int - 2 bytes)
        #   - next 1 code data type (08 = unsigned byte) (B - unsigned char - 1 byte)
        #   - last 1 code dimensions for vector/matrix (B - unsigned char - 1 byte)
        zero, type_data, dimensions = struct.unpack('>HBB', f.read(4))
        print("Data type:", type_data)
        print("Data dimensions:", dimensions)
        
        # For N dimensions above, the next N * 4 bytes (32 bit unsigned ints)
        #   are sizes of each dimension.
        # Can use read(4) again as previous use leaves file pointer 
        #    with a 4 byte offset. seek(pos) can be used to manually change pos
        dim_sizes = []
        for i in range(dimensions):
            dim_sizes.append(struct.unpack('>I', f.read(4))[0])
        print("Size of each dimension:", dim_sizes, "\n")

        # list is the shape of output. read() will now read to EOF.
        #  As unsigned byte, use np.uint8 dtype 
        #   fromstring gives us single 1-D list of all remaining bytes
        byte_arr =  np.fromstring(f.read(), dtype=np.uint8)
        
    # reshape 1-D bytestream array as described above
    shape = tuple(dim_sizes)
    return byte_arr.reshape(shape)
