import random

def binary_search(sorted_list, target):

    sorted_list.sort()

    if len(sorted_list) == 0:
        return 'Value not found. Value not in list'
    mid_idx = len(sorted_list) // 2
    mid_val = sorted_list[mid_idx]
    if mid_val == target:
        return mid_idx
    if mid_val > target:
        left_half = sorted_list[:mid_idx]
        return binary_search(left_half, target)
    if mid_val < target:
        right_half = sorted_list[mid_idx + 1:]
        result = binary_search(right_half, target)
        if result == "Value not found. Value not in list":
            return result
        else:
            return result + mid_idx + 1

def partition(list, start, end):
    follower = leader = start
    while leader < end:
        if list[leader] <= list[end]:
            list[follower], list[leader] = list[leader], list[follower]
            follower += 1
        leader += 1
    list[follower], list[end] = list[end], list[follower]
    return follower

def quicksort_helper(list, start, end):
    if start >= end:
        return
    p = partition(list, start, end)
    quicksort_helper(list, start, p-1)
    quicksort_helper(list, p +1, end)

def quicksort(list):
    return quicksort_helper(list, 0, len(list) - 1)

def random_list():
    list = [random.randrange(50) for i in range(50)]
    print(list)
    return list

def main():
    list = random_list()
    quicksort(list)
    print(list)

if __name__ == '__main__':
    main()
