import os

if __name__ == '__main__':
    # go through all output folders and look in each folder for the python file output/{folder}/{folder}.py
    # run the python by importing the function test_for_accuracy


    # get all folders in output
    output_folders = os.listdir("output")
    for name in output_folders:
        package = __import__(f"output.{name}.{name}")
        test_for_accuracy = getattr(getattr(getattr(package, name), name), "test_for_accuracy")

        print(f"{test_for_accuracy()}")





