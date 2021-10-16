channels1 = 8
kernal1 = 5
stride1 = 1
pad1 = 0 #int(kernal1 / 2)
pool1 = 2

pool_width = (1 + (28 + 2 * pad1 - kernal1) / stride1) / pool1
none_width = (1 + (28 + 2 * pad1 - kernal1) / stride1)
print(f'pool: {pool_width}')
print(f'none: {none_width}')

channels2 = 16
kernal2 = 3
stride2 = 1
pad2 = 0 #int(kernal2 / 2)
pool2 = 2

pool_pool_width = (1 + (pool_width + 2 * pad2 - kernal2) / stride2) / pool2
pool_none_width = (1 + (pool_width + 2 * pad2 - kernal2) / stride2)
none_pool_width = (1 + (none_width + 2 * pad2 - kernal2) / stride2) / pool2
none_none_width = (1 + (none_width + 2 * pad2 - kernal2) / stride2)

print(f'pool pool: {pool_pool_width}')
print(f'pool none: {pool_none_width}')
print(f'none pool: {none_pool_width}')
print(f'none none: {none_none_width}')

size1 = channels2 * (pool_pool_width ** 2)
size2 = channels2 * (pool_none_width ** 2)
size3 = channels2 * (none_pool_width ** 2)
size4 = channels2 * (none_none_width ** 2)

print(size1)
print(size2)
print(size3)
print(size4)

full = 150
