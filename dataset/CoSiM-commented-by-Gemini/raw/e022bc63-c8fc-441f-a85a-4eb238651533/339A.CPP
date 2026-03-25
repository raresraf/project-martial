
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

/**
 * @file 339A.CPP
 * @brief Rearranges a sum of numbers 1, 2, and 3 into non-decreasing order.
 *
 * @details
 * Problem Statement:
 * Xenia, a beginner mathematician, can only calculate a sum if the summands
 * are in non-decreasing order. Given a string representing a sum of numbers
 * 1, 2, and 3 (e.g., "1+3+2"), the task is to rearrange the summands and print
 * the sum in an order that Xenia can calculate (e.g., "1+2+3").
 *
 * Input:
 * A non-empty string `s` representing the sum. It contains only digits
 * ('1', '2', '3') and '+' characters, with a maximum length of 100.
 *
 * Output:
 * The rearranged sum.
 *
 * Algorithm:
 * This solution follows a standard Extract-Sort-Reconstruct approach.
 * 1.  **Extract**: The numeric values (as integers) are extracted from the
 *     input string and stored in a `std::vector`. This is done by iterating
 *     through the string with a step of 2 to skip the '+' characters.
 * 2.  **Sort**: The `std::vector` of numbers is sorted in non-decreasing
 *     order using `std::sort`.
 * 3.  **Reconstruct**: The sorted vector is iterated through to build the
 *     output string, inserting a '+' separator between the numbers.
 *
 * Time Complexity: O(L log L), where L is the number of summands, dominated
 * by the sort operation.
 * Space Complexity: O(L) to store the numbers in a vector.
 */

using namespace std;

int main() {
  string s;
  vector<int> v;
  cin >> s;

  // Block 1: Extract the numbers from the input string.
  // The loop increments by 2 to skip the '+' signs.
  for (int i = 0; i < s.length(); i += 2) {
    // Convert character digit to integer by subtracting the ASCII value of '0' (48).
    v.push_back(s[i] - 48);
  }

  // Block 2: Sort the vector of numbers in non-decreasing order.
  sort(v.begin(), v.end());

  // Block 3: Reconstruct and print the sorted sum.
  // Invariant: The loop has printed the first `it - v.begin()` elements,
  // correctly separated by '+'.
  for (vector<int>::iterator it = v.begin(); it != v.end(); it++) {
    // Pre-condition: Check if this is the first element.
    if (it == v.begin()) {
      // Don't print a '+' before the first number.
      cout << *it;
    } else {
      // Print a '+' before all subsequent numbers.
      cout << "+" << *it;
    }
  }
  cout << endl;
  return 0;
}
