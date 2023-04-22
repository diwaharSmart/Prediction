class Guess:

    # Initializing the parameter
    def __init__(self, processing_data, previous_data):
        self.processing_data = [list(d) for d in processing_data]
        self.previous_data = [list(d) for d in previous_data]
        self.pattern = {"A": ['0', '1'], "B": ['2', '3'], "C": ['4', '5'], "D": ['6', '7'], "E": ['8', '9']}
        self.all_numbers = [str(i) for i in range(0, 10)]
        self.previous_data_with_pattern = [[next((key for key, value in self.pattern.items() if char in value), '') for char in num] for num in previous_data]
        self.processing_data_with_pattern = [[next((key for key, value in self.pattern.items() if char in value), '') for char in num] for num in processing_data]

    def last_three_left_and_right_comparision(self):
        first = self.processing_data[-1]
        second = self.processing_data[-2]
        third = self.processing_data[-3]

        # print(first,second,third)

        left_half = []
        left_half.extend(first[:3])
        left_half.extend(second[:3])
        left_half.extend(third[:3])
        left_missing_numbers = set(self.all_numbers).difference(set(left_half))

        right_half = []
        right_half.extend(first[3:])
        right_half.extend(second[3:])
        right_half.extend(third[3:])
        right_half_missing_number = set(self.all_numbers).difference(set(right_half))
        left_right_intersection   = set(left_half).intersection(set(right_half))
        left_right_difference     = set(left_half).difference(set(right_half))
        right_left_difference     = set(right_half).difference(set(left_half))

        common_difference =   left_right_difference.intersection(right_left_difference)

        full_missing = set(left_missing_numbers).intersection(set(right_half_missing_number))

        res = dict()
        suspisious = list(left_missing_numbers)+ list(right_half_missing_number)+ list(left_right_difference)+ list(right_left_difference)+list(common_difference)+list(full_missing)
        res["out"] = [left_right_intersection,set(self.all_numbers).difference(set(suspisious)), left_missing_numbers, right_half_missing_number, left_right_difference, right_left_difference, common_difference, full_missing ]
        res["left_missing_numbers"] = left_missing_numbers
        res["right_missing_numbers"] = right_half_missing_number
        res["left_right_difference"] = left_right_difference
        res["right_left_difference"] = right_left_difference
        res["left_right_difference_intersection"] = common_difference
        res["left_right_intersection"] = left_right_intersection
        res["full_missing"] = full_missing

        print(res)


    def first_two_numbers_came_last_two_days(self):
        res = {"day_1":None,"day_2":None,"day_3":None,"day_4":None}
        pass

