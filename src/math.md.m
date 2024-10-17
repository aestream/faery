

alpha_o = a + b * (1 - a)

c_o = (c_a * a + c_b * b * (1 - a)) / (a + b * (1 - a))


c_a: text color
c_b: background color
a: text transparency
b: background transparency
t: bitmap
e: text color

c_o@c_a=1&b=1 = (a + c_b * (1 - a))



c_b * (255 - a) / 255 + a

Integer version:

c_o = 255 * (c_a / 255 * a / 255 + c_b / 255 * b / 255 * (1 - a / 255)) / (a / 255 + b / 255 * (1 - a / 255))
    = 255 * (255 * 255 * [c_a / 255 * a / 255 + c_b / 255 * b / 255 * (1 - a / 255)]) / (255 * 255 * [a / 255 + b / 255 * (1 - a / 255)])
    = ...                                                                             / (255 * a + (255 - a) * b)
    = (255 * c_a * a + (255 - a) * c_b * b)                                           / ...
    = (255 * c_a * a + (255 - a) * c_b * b) / (255 * a + (255 - a) * b)

a = bitmap * text_color / 255 = t * e / 255

c_o = (255 * c_a * t * e / 255 + (255 - t * e / 255) * c_b * b) / (255 * t * e / 255 + (255 - t * e / 255) * b)
c_o = (255 * c_a * t * e + (255² - t * e) * c_b * b) /  (255 * t * e + (255² - t * e) * b)



(255 * a * c_a + (255 - a) * c_b * b)


(255 * c_a * t * e + (255² - t * e) * c_b * b)

t * e * [255 * c_a + (255² / (t * e) - 1)]
