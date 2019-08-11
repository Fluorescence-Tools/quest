class MFM():
    
    def __init__(self):
        pass

import zerorpc
s = zerorpc.Server(MFM())
s.bind("tcp://0.0.0.0:4242")
s.run()
