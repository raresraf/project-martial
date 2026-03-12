/**
 * @59490ef5-bc76-4606-8e49-e3eb7514fa84/160A.CPP
 * @brief This program reads a series of integers, calculates their total sum,
 * and sorts them. It appears to be an incomplete solution to a problem
 * that likely requires selecting a subset of these numbers, possibly to
 * maximize a sum or satisfy a specific condition.
 * The core logic for the problem's objective is missing in the final loop.
 *
 * Algorithm (Implemented part): Input reading, summation, sorting, and an empty reverse iteration.
 * Time Complexity (Implemented part): O(N log N) due to sorting, where N is the number of integers.
 * Space Complexity (Implemented part): O(N) for storing the integers in a vector.
 */

#include <bits/stdc++.h> // A common header in competitive programming; includes iostream, vector, algorithm.
using namespace std;     // Uses the standard namespace.

int main() {
  int n, sum = 0; // Declares 'n' for the count of integers, 'sum' initialized to 0.
  cin >> n;       // Reads the number of integers from standard input.
  vector<int> v;  // Declares a dynamic array (vector) to store the integers.
  
  /**
   * Block Logic: Reads 'n' integers from input, calculates their sum, and stores them in a vector.
   * Invariant: After this loop, 'v' contains all 'n' integers and 'sum' holds their total.
   */
  for (int i = 0; i < n; i++) {
    int tmp;    // Temporary variable to store each integer read.
    cin >> tmp; // Reads an integer.
    sum += tmp; // Adds the integer to the total sum.
    v.push_back(tmp); // Adds the integer to the vector.
  }

  // Block Logic: Sorts the vector of integers in ascending order.
  // Pre-condition: 'v' contains 'n' integers.
  // Invariant: 'v' is sorted in non-decreasing order.
  sort(v.begin(), v.end());

  /**
   * Block Logic: This loop is intended for further processing, likely to select elements
   * from the sorted vector (starting with the largest). However, the loop body is empty,
   * indicating that the problem's main logic is incomplete.
   * Invariant: No operations are performed within this loop in the current implementation.
   */
  for (auto i = v.end(); i != v.begin(); --i) {
    // Missing logic here. For example, in problems like "partition into two sets
    // such that one set's sum is greater than another", one would pick elements
    // from the end of the sorted array.
  }

  return 0; // Indicates successful program termination.
}