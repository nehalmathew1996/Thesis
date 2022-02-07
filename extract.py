import cv2


def number_plate_extract(x_min,y_min,x_max,y_max):

    x = int(x_min)
    y = int(y_min)
    w = int(x_max - x_min)
    h = int(y_max - y_min)

    img = 'image.jpg'

    img = cv2.imread(img)
    print(type(img))

    number_plate = img[y:y+h,x:x+w]

    cv2.imwrite('number_plate.jpg', number_plate)

    cv2.destroyAllWindows()

