import pandas as pd

from predictor.guessings import Guess
from utils.input_data import Data


def convert_to_csv(file_name,file_dir):
    # file_name = input("Output File Name : ")
    # file_dir  = input("Image Path : ")

    data = Data(soucre_file=None)
    img = data.import_image_file(file_dir=file_dir)

    inverted_image = data.invert_image(img=img)
    inverted_image_with_otsu = data.inverted_image_with_otsu(img=img)

    kernel = data.create_rectangular_kernal()

    vertical_lines = data.perform_erosion_vertical(img_bin_otsu=inverted_image_with_otsu, img=img)
    horizontal_lines = data.perform_erosion_horizontal(img_bin=inverted_image, img=img)

    add_two_images = data.add_two_images(vertical_lines=vertical_lines, horizontal_lines=horizontal_lines,
                                         kernel=kernel, img=img)

    get_contours_and_bounding_boxes = data.get_contours_and_bounding_boxes(
        vertical_horizontal_lines=add_two_images["vertical_horizontal_lines"], img=img)

    store_rows_and_coloumns = data.store_rows_and_coloumns(boxes=get_contours_and_bounding_boxes["boxes"],
                                                           boundingBoxes=get_contours_and_bounding_boxes[
                                                               "boundingBoxes"])

    create_box_coordinate_list = data.create_box_coordinate_list(rows=store_rows_and_coloumns["rows"],
                                                                 total_cells=store_rows_and_coloumns["total_cells"],
                                                                 center=store_rows_and_coloumns["center"])

    dataframe_final = data.extract_boxes_using_pytesseract(boxes_list=create_box_coordinate_list,
                                                           bitnot=add_two_images["bitnot"])

    df = data.create_dataframe(dataframe_final=dataframe_final, rows=store_rows_and_coloumns["rows"],
                               total_cells=store_rows_and_coloumns["total_cells"])

    # print(df)
    df.to_csv('file/excel/'+str(file_name)+'.csv')
    print(file_name)
    print("Completed")
    return None
# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    data = Data(soucre_file=None)
    previous_data = data.load_data()
    """
    # convert_to_csv(out[i],file_paths[i])
    # data = Data(soucre_file="file/excel/*.csv")
    # new_data = data.convert_excel_to_list()
    # data.save_data(new_data)
    # data.update_data(["215003"])
    # data.remove_last_data()
    """
    # print(previous_data[-5:])
    # data.update_data(["215003"])
    print(previous_data[-3:])
    guess = Guess(previous_data=previous_data, processing_data=previous_data[-3:])
    guess.last_three_left_and_right_comparision()




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
