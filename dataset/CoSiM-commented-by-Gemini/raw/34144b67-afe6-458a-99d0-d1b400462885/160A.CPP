/**
 * @file 160A.CPP
 * @brief Counts the number of pairs of integers that satisfy a specific condition.
 *
 * This program reads a number of test cases, T. For each test case, it reads
 * two integers, `manus` and `capacity`. It then counts how many of these pairs
 * satisfy the condition `capacity - manus >= 2`.
 *
 * Algorithm:
 * 1. Read the total number of test cases (T).
 * 2. Initialize a counter `n` to 0.
 * 3. Loop T times:
 *    a. Read the two integers for the current test case.
 *    b. Check if the difference between them is 2 or more.
 *    c. If it is, increment the counter `n`.
 * 4. Print the final count.
 *
 * Time Complexity: O(T), where T is the number of test cases, as each test case
 * involves a constant number of operations.
 * Space Complexity: O(1), as the program uses a fixed amount of memory
 * regardless of the input size.
 */
#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  // manus: Likely represents a used amount or a starting value.
  // capacity: Represents the total capacity or an upper limit.
  // T: The number of test cases.
  // n: A counter for test cases that meet the condition.
  // v: The calculated difference between capacity and manus.
  int manus, capacity, T, n = 0, v;
  
  // Read the number of test cases.
  cin >> T;

  /**
   * Block Logic: Main processing loop.
   * Iterates T times, processing one pair of integers in each iteration.
   */
  while (T--) {
    cin >> manus >> capacity;
    
    // Calculate the available room or difference.
    v = capacity - manus;

    // Pre-condition: Check if there is enough "room" (at least 2 units).
    if (v >= 2) {
      n++;
    }
  }

  // Output the total count of pairs that satisfied the condition.
  cout << n << endl;
  return 0;
}
