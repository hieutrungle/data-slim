# Huffman Coding in python
from collections import Counter
import sys
import copy
import numpy as np

# Creating tree nodes
class NodeTree:
    def __init__(self, freq=None, left=None, right=None, is_leaf=None, value=None):
        self.freq = freq
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.value = value

    def children(self):
        return (self.left, self.right)

    def __str__(self):
        # return '%s_%s' % (self.left, self.right)
        s = "%s_%s" % (self.left, self.right)
        return s


class Huffman:
    def __init__(self):
        self.hufftable = {}

        self.huffman_table = {}
        self.codes = []
        self.leaf_freqs = []

    def huffman_code_tree(self, node, left=True, binString=""):
        # if isinstance(node, str) or isinstance(node, int) or isinstance(node, np.):
        #     return {node: binString}
        if not isinstance(node, NodeTree):
            return {node: binString}
        if node.is_leaf == True:
            return {node.value: binString}
        (left, right) = node.children()
        d = dict()
        d.update(self.huffman_code_tree(left, True, binString + "0"))
        d.update(self.huffman_code_tree(right, False, binString + "1"))

        return d

    def _huffmanCodes(self, tl):
        if tl.is_leaf == False:
            l = tl.left
            r = tl.right
            self.codes.append("0")
            self._huffmanCodes(l)
            self.codes.pop()
            self.codes.append("1")
            self._huffmanCodes(r)
            self.codes.pop()
        else:
            self.hufftable[tl.value] = "".join(self.codes)

    def _get_sorted_nodes(self, data):
        # sort in decending order
        self.leaf_freqs = dict(Counter(data))
        self.leaf_freqs = sorted(
            self.leaf_freqs.items(), key=lambda x: x[1], reverse=True
        )
        nodes = copy.deepcopy(self.leaf_freqs)
        return nodes

    def print_huffman_table(self, freqs, huffman_table):
        print(" Char | Huffman code ")
        print("----------------------")
        for (char, _) in freqs:
            print(" %-4r |%12s" % (char, huffman_table[char]))

    def encode_numbers(self, data, verbose=False):

        self.tf = dict(Counter(data))
        leaf_freqs = sorted(self.tf.items(), key=lambda x: x[1], reverse=True)

        lk = self.tf.keys()

        tl = [NodeTree(self.tf.get(k), None, None, True, k) for k in lk]
        tl.sort(key=lambda x: x.freq)

        while len(tl) > 1:
            l = tl.pop(0)
            r = tl.pop(0)
            tl.append(NodeTree(l.freq + r.freq, l, r, False, None))
            tl.sort(key=lambda x: x.freq)

        self._huffmanCodes(tl.pop(0))
        self.print_huffman_table(leaf_freqs, self.hufftable)
        ###

        nodes = self._get_sorted_nodes(data)

        while len(nodes) > 1:
            (node1, freq1) = nodes.pop()
            (node2, freq2) = nodes.pop()
            sum_freq = freq1 + freq2
            new_node = NodeTree(freq1 + freq2, node1, node2)
            nodes.append((new_node, sum_freq))
            nodes = sorted(nodes, key=lambda x: x[1], reverse=True)

        # print(nodes[0][0])
        self.huffman_table = self.huffman_code_tree(nodes[0][0])
        self.huffman_table = dict(
            sorted(self.huffman_table.items(), key=lambda x: len(x[1]))
        )

        self.print_huffman_table(self.leaf_freqs, self.huffman_table)

        compressed_data = "".join([self.huffman_table[b] for b in data])
        # print(f"Compressed data: {compressed_data}")

        header_code = self.__encode_numbers_hufftable()
        compressed_data = header_code + compressed_data

        num_bin_added = 0
        while not len(compressed_data) % 8 == 0:
            compressed_data += "0"
            num_bin_added += 1

        num_bin_added = format(num_bin_added, "08b")
        # print(f"Header code: {header_code}")
        # print(f"Binary added: {num_bin_added}")
        print(
            f"final compressed data = num_bin_added + header_code "
            f"+ compressed_data + trailing_zeros"
        )

        compressed_data = num_bin_added + compressed_data

        if verbose:
            max_symbol = max(self.huffman_table.keys())
            bin_max_symbol = len(bin(max_symbol)) - 2
            bin_string = [np.binary_repr(num, width=bin_max_symbol) for num in data]
            bin_string = "".join(bin_string)
            print(f"Compression ratio: {len(bin_string) / len(compressed_data)}")

        return compressed_data

    def __encode_numbers_hufftable(self):

        encoding_table = ""

        # header 1: max symbol
        max_symbol = max(self.huffman_table.keys())
        bin_max_symbol = len(bin(max_symbol)) - 2  # -2 because of 0b format of binary

        # header 2: min length code
        len_huffman_codes = list(set([len(v) for v in self.huffman_table.values()]))
        len_huffman_codes.sort(reverse=False)
        min_len_code = min(len_huffman_codes)
        bin_min_len_code = len(bin(min_len_code)) - 2

        # header 3: max length of len_huffman_codes differences
        len_diffs = self.__get_diffs_lengths(len_huffman_codes)
        max_len_diff = max(len_diffs)
        bin_max_len_diff = len(bin(max_len_diff)) - 2

        # print(f"bin_max_symbol: {bin_max_symbol}")
        # print(f"bin_min_len_code: {bin_min_len_code}")
        # print(f"bin_max_len_diff: {bin_max_len_diff}")

        # header 4: get lenghts of huffman table
        # get ascending order of huffman_table
        data_huffcodes = self.__data_inv_huffcodes()
        data_huffman_codes = sorted(data_huffcodes, key=lambda t: t[2])

        # get binary of huffman table: symbol + len(code) + code
        symbol = format(data_huffman_codes[0][0], "0" + str(bin_max_symbol) + "b")
        len_code = format(min_len_code, "0" + str(bin_min_len_code) + "b")
        code = data_huffman_codes[0][1]  # already in binary
        # print(f"symbol: {symbol}; len_code: {len_code}; code: {code}")
        encoding_table += symbol + len_code + code
        last_len = data_huffman_codes[0][2]

        for i in range(1, len(data_huffman_codes)):
            symbol = format(data_huffman_codes[i][0], "0" + str(bin_max_symbol) + "b")
            if data_huffman_codes[i][2] > last_len:
                len_code = format(
                    data_huffman_codes[i][2] - last_len,
                    "0" + str(bin_max_len_diff) + "b",
                )
                last_len = data_huffman_codes[i][2]
            else:
                len_code = format(0, "0" + str(bin_max_len_diff) + "b")

            code = data_huffman_codes[i][1]
            encoding_table += symbol + len_code + code
            # print(f"symbol: {symbol}; len_code: {len_code}; code: {code}")

        # encoding_tabl1 = ""
        # for i, (symbol, value) in enumerate(self.huffman_table.items()):
        #     symbol = format(symbol, "0" + str(bin_max_symbol) + "b")
        #     len_code = min_len_code if i < len(self.huffman_table) - 1 else 0
        #     len_code = format(len_code, "0" + str(bin_min_len_code) + "b")
        #     code = value  # already in binary
        #     encoding_tabl1 += symbol + len_code + code
        #     print(f"symbol: {symbol}; len_code: {len_code}; code: {code}")

        # if encoding_table == encoding_tabl1:
        #     print("s1 and s2 are equal.")
        # else:
        #     print("s1 and s2 are not equal.")

        # header binary
        header = format(bin_max_symbol, "08b")
        header += format(bin_min_len_code, "08b")
        header += format(bin_max_len_diff, "08b")
        header += format(len(data_huffman_codes), "08b")

        return header + encoding_table

    def __data_inv_huffcodes(self):
        return [(k, v, len(v)) for k, v in self.huffman_table.items()]

    def decode(self, datac):

        ht, datac = self.__decode_hufftable(datac)
        lengths = self.__sorted_lengths_by_frequency(ht)

        datad = []
        c_size = len(datac)

        index = 0
        while index < c_size:
            for left in lengths:
                possible_code = datac[index : index + left]
                if possible_code in ht.keys():
                    datad.append(ht[possible_code])
                    index = index + left
                    break

        return datad

    def __decode_hufftable(self, datac):

        huffman_table = {}
        data_header = []
        header = datac[0:40]
        datac = datac[40:]

        for i in range(0, 40, 8):
            data_header.append(int(header[i : i + 8], 2))

        rem = data_header[0]
        max_symbol = data_header[1]
        min_len_code = data_header[2]
        max_len_diff = data_header[3]
        nc = data_header[4]

        i = 0
        last_length = 0

        k = int(datac[0:max_symbol], 2)
        datac = datac[max_symbol:]
        length = int(datac[0:min_len_code], 2)
        datac = datac[min_len_code:]
        last_length += length
        v = datac[0:last_length]
        datac = datac[last_length:]
        huffman_table[v] = k

        while i < nc - 1:
            k = int(datac[0:max_symbol], 2)
            datac = datac[max_symbol:]
            length = int(datac[0:max_len_diff], 2)
            datac = datac[max_len_diff:]
            last_length += length
            v = datac[0:last_length]
            datac = datac[last_length:]
            huffman_table[v] = k
            i += 1

        if rem > 0:
            datac = datac[:-rem]

        return huffman_table, datac

    def __get_diffs_lengths(self, lengths):
        return [lengths[i + 1] - lengths[i] for i in range(len(lengths) - 1)]

    def __sorted_lengths_by_frequency(self, ht):

        lengths = list(set([len(k) for k in ht.keys()]))
        lengths.sort(reverse=False)
        return lengths


if __name__ == "__main__":
    rng = np.random.default_rng(12345)
    int_lists = rng.integers(low=0, high=128, size=5000)
    # # int_lists = [int(i) for i in int_lists]
    # print(int_lists)

    # Encoding
    huffman = Huffman()
    compressed_data = huffman.encode_numbers(int_lists, verbose=True)
    # print(f"Compressed data: {compressed_data}")
    # print(np.binary_repr(int_lists))
    # bin_string = [np.binary_repr(int_num, width=8) for int_num in int_lists]
    # bin_string = "".join(bin_string)
    # print(f"Binary lists: {bin_string}")
    # '{:08b}'.format(a[i])

    huffman_dec = Huffman()
    data = huffman_dec.decode(compressed_data)
    # print(f"decoded data: {data}")
