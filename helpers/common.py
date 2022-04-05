# currently only used in deprecated function


# helper function: converts a byte to integer
def bytes_to_int(byte_data: bytes) -> int:
    return int.from_bytes(byte_data, 'big')


# helper function: converts a byte to character
def bytes_to_char(byte_data: bytes) -> str:
    integer = ord(byte_data)
    string = chr(integer + 64)

    return string
