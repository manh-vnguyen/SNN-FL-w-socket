import multiprocessing

def worker(data):
    # Modify the data object in the child process
    data["number"] += 1
    print(f"Inside Child Process: {data}")

if __name__ == "__main__":
    # Original data in parent process
    original_data = {"number": 0}

    # Print original data before modification in child process
    print(f"Before Child Process: {original_data}")

    # Create and start the process
    p = multiprocessing.Process(target=worker, args=(original_data,))
    p.start()
    p.join()

    # Print original data after modification attempt in child process
    print(f"After Child Process: {original_data}")
