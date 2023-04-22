import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract


class Data:

    def __init__(self, soucre_file=None):
        self.source_file = soucre_file

    @staticmethod
    def separate_numbers(data):
        # Convert each string in the list to an integer and create a list of lists
        numbers_list = [list(map(int, s)) for s in data]
        return numbers_list

    @staticmethod
    def import_image_file(file_dir):
        file = file_dir
        img = cv2.imread(file, 0)
        return img

    @staticmethod
    def invert_image(img):
        thresh, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        img_bin = 255 - img_bin
        # plotting = plt.imshow(img_bin, cmap='gray')
        # plt.title("Inverted Image with global thresh holding")
        # plt.show()
        return img_bin

    @staticmethod
    def inverted_image_with_otsu(img):
        img_bin1 = 255 - img
        thresh1, img_bin1_otsu = cv2.threshold(img_bin1, 128, 255, cv2.THRESH_OTSU)
        # plotting = plt.imshow(img_bin1_otsu, cmap='gray')
        # plt.title("Inverted Image with otsu thresh holding")
        # plt.show()
        return img_bin1_otsu

    @staticmethod
    def create_rectangular_kernal():
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return kernel

    @staticmethod
    def perform_erosion_vertical(img_bin_otsu, img):
        plt.figure(figsize=(30, 30))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, np.array(img).shape[1] // 100))
        eroded_image = cv2.erode(img_bin_otsu, vertical_kernel, iterations=3)
        # plt.subplot(151), plt.imshow(eroded_image, cmap='gray')
        # plt.title('Image after erosion with vertical kernel'), plt.xticks([]), plt.yticks([])
        vertical_lines = cv2.dilate(eroded_image, vertical_kernel, iterations=3)
        # plt.subplot(152), plt.imshow(vertical_lines, cmap='gray')
        # plt.title('Image after dilation with vertical kernel'), plt.xticks([]), plt.yticks([])
        #
        # plt.show()
        return vertical_lines

    @staticmethod
    def perform_erosion_horizontal(img_bin, img):
        plt.figure(figsize=(30, 30))
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (np.array(img).shape[1] // 100, 1))
        horizontal_lines = cv2.erode(img_bin, hor_kernel, iterations=5)
        # plt.subplot(153), plt.imshow(horizontal_lines, cmap='gray')
        # plt.title('Image after erosion with horizontal kernel'), plt.xticks([]), plt.yticks([])
        horizontal_lines = cv2.dilate(horizontal_lines, hor_kernel, iterations=5)
        # plt.subplot(154), plt.imshow(horizontal_lines, cmap='gray')
        # plt.title('Image after dilation with horizontal kernel'), plt.xticks([]), plt.yticks([])
        #
        # plt.show()

        return horizontal_lines

    @staticmethod
    def add_two_images(vertical_lines, horizontal_lines, kernel, img):

        vertical_horizontal_lines = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
        vertical_horizontal_lines = cv2.erode(~vertical_horizontal_lines, kernel, iterations=3)

        thresh, vertical_horizontal_lines = cv2.threshold(vertical_horizontal_lines, 128, 255,
                                                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        bitxor = cv2.bitwise_xor(img, vertical_horizontal_lines)

        bitnot = cv2.bitwise_not(bitxor)

        return {"bitxor": bitxor, "bitnot": bitnot, "vertical_horizontal_lines": vertical_horizontal_lines}

    @staticmethod
    def get_contours_and_bounding_boxes(vertical_horizontal_lines, img):
        contours, hierarchy = cv2.findContours(vertical_horizontal_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        boundingBoxes = [cv2.boundingRect(contour) for contour in contours]
        (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda x: x[1][1]))

        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w < 1000 and h < 500:
                # image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                boxes.append([x, y, w, h])

        return {"boxes": boxes, "boundingBoxes": boundingBoxes}

    @staticmethod
    def store_rows_and_coloumns(boxes, boundingBoxes):
        rows = []
        columns = []
        heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
        mean = np.mean(heights)
        # print(mean)
        columns.append(boxes[0])
        previous = boxes[0]
        for i in range(1, len(boxes)):
            if boxes[i][1] <= previous[1] + mean / 2:
                columns.append(boxes[i])
                previous = boxes[i]
                if i == len(boxes) - 1:
                    rows.append(columns)
            else:
                rows.append(columns)
                columns = []
                previous = boxes[i]
                columns.append(boxes[i])
        print("Rows")
        total_cells = 0
        print(rows)
        # for row in rows:
        #     # print(row)

        for i in range(len(rows)):
            if len(rows[i]) > total_cells:
                total_cells = len(rows[i])

        center = [int(rows[i][j][0] + rows[i][j][2] / 2) for j in range(len(rows[i])) if rows[0]]
        center = np.array(center)
        center.sort()
        print(total_cells, len(rows), center)

        return {"rows": rows, "columns": columns, "total_cells": total_cells, "center": center}

    @staticmethod
    def create_box_coordinate_list(rows, total_cells, center):

        boxes_list = []
        for i in range(len(rows)):
            l = []
            # print(total_cells)
            for k in range(total_cells):
                l.append([])
            for j in range(len(rows[i])):
                diff = abs(center - (rows[i][j][0] + rows[i][j][2] / 4))
                minimum = min(diff)
                indexing = list(diff).index(minimum)
                l[indexing].append(rows[i][j])
            boxes_list.append(l)

        return boxes_list

    @staticmethod
    def extract_boxes_using_pytesseract(boxes_list, bitnot):
        dataframe_final = []
        for i in range(len(boxes_list)):
            for j in range(len(boxes_list[i])):
                s = ''
                if len(boxes_list[i][j]) == 0:
                    dataframe_final.append(' ')
                else:
                    for k in range(len(boxes_list[i][j])):
                        y, x, w, h = boxes_list[i][j][k][0], boxes_list[i][j][k][1], boxes_list[i][j][k][2], \
                                     boxes_list[i][j][k][3]
                        roi = bitnot[x:x + h, y:y + w]
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                        border = cv2.copyMakeBorder(roi, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
                        resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                        dilation = cv2.dilate(resizing, kernel, iterations=1)
                        erosion = cv2.erode(dilation, kernel, iterations=2)
                        out = pytesseract.image_to_string(erosion)
                        if len(out) == 0:
                            out = pytesseract.image_to_string(erosion)
                        s = s + " " + out
                    dataframe_final.append(s)

        return dataframe_final

    @staticmethod
    def create_dataframe(dataframe_final, rows, total_cells):

        arr = np.array(dataframe_final)
        dataframe = pd.DataFrame(arr.reshape(len(rows), total_cells))

        return dataframe

    def convert_excel_to_list(self):
        import glob

        # Get a list of file paths that match the pattern "*.csv" in the directory
        csv_files = glob.glob(self.source_file)

        # Create an empty list to hold the dataframes
        dfs = []
        # Loop over the file paths and read each file into a dataframe
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        column_lists = []
        for df in dfs:
            for _, column_data in df.iloc[:, 1:].items():
                column_list = column_data.tolist()
                column_lists.extend(column_list)

        # print(column_lists)
        cleaned_column_list = [str(s).replace('\x0c', '').replace('\n', '').replace(" ", "") for s in column_lists]
        removed_empty_strings = [s for s in cleaned_column_list if s]
        return removed_empty_strings

    @staticmethod
    def save_data(data):
        import pickle
        with open('data.p', 'wb') as f:
            # Use pickle.dump() to save the list as bytes
            pickle.dump(data, f)

    @staticmethod
    def load_data():
        import pickle
        with open('data.p', 'rb') as f:
            # Use pickle.load() to load the pickled data
            data = pickle.load(f)
        return data

    @staticmethod
    def update_data(new):
        import pickle
        with open('data.p', 'rb') as f:
            # Use pickle.load() to load the pickled data
            data = pickle.load(f)
        data.extend(new)
        with open('data.p', 'wb') as f:
            # Use pickle.dump() to save the list as bytes
            pickle.dump(data, f)
        return data

    @staticmethod
    def remove_last_data():
        import pickle
        with open('data.p', 'rb') as f:
            # Use pickle.load() to load the pickled data
            data = pickle.load(f)

        del data[-1]
        with open('data.p', 'wb') as f:
            # Use pickle.dump() to save the list as bytes
            pickle.dump(data, f)
        return data
