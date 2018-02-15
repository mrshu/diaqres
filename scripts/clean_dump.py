import re
import sys
TAGS = re.compile(r'<[^>]+>', re.IGNORECASE)
DEBREE = re.compile(r'\%.+>', re.IGNORECASE)
TABLE_DEBREE = re.compile(r'^\![^\|]+\|', re.IGNORECASE)
PX_DEBREE = re.compile(r'px[^>]+>', re.IGNORECASE)


def clean_dump(filename):
    outfile = '{}.clean'.format(filename)
    with open(filename, 'r') as f:
        with open(outfile, 'w') as out:
            for line in f:
                line = TAGS.sub('', line)
                line = DEBREE.sub('', line)
                line = TABLE_DEBREE.sub('', line)
                line = PX_DEBREE.sub('', line)
                out.write(line)

if __name__ == "__main__":
    clean_dump(sys.argv[1])
