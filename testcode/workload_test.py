import multiprocessing
import time

def fib(n):
    """Recursively calculates the nth Fibonacci number."""
    if n <= 1:
        return n
    else:
        return fib(n-1) + fib(n-2)

def worker(proc_num, fib_number):
    print(f"Process {proc_num}: calculating Fibonacci({fib_number})")
    start_time = time.time()
    result = fib(fib_number)
    end_time = time.time()
    print(f"Process {proc_num}: Fibonacci({fib_number}) = {result}. Calculation took {end_time - start_time} seconds.")

def main():
    fib_number = 35  # Adjust this number to increase or decrease the workload
    num_processes = multiprocessing.cpu_count()  # Number of CPU cores
    print(f"Starting workload test with {num_processes} processes.")

    processes = []
    for i in range(num_processes):
        p = multiprocessing.Process(target=worker, args=(i, fib_number))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()  # Wait for all processes to complete

    print("Workload test completed.")

if __name__ == "__main__":
    main()
