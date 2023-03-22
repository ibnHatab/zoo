

camel = [
    "_gierwnkl",
    "_gierrate",
    "_positionx",
    "_positiony",
    "_positionz",
]

snake = [
    "_gier_wnkl",
    "_gier_rate",
    "_position_x",
    "_position_y",
    "_position_z",
]

camel_map = [(x, x.replace("_", "")) for x in camel]
snake_map = [(x, x.replace("_", "")) for x in snake]

camel_map = camel_map.sort(key=lambda x: x[1], reverse=True)
snake_map = snake_map.sort(key=lambda x: x[1], reverse=True)

for c, s in zip(camel_map, snake_map):
    assert c[1] == s[1]
    print(c[0], s[0])
