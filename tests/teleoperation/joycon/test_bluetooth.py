import hid
devices = hid.enumerate()
for dev in devices:
    print(dev)