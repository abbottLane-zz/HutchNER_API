# Copyright (c) 2016-2017 Fred Hutchinson Cancer Research Center
#
# Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
from Sectioning.ClinicalSectioner import uw_sectioner

def main(documents):
    # Section raw documents
    return uw_sectioner(documents)

if __name__ == '__main__':
    main(documents=None)