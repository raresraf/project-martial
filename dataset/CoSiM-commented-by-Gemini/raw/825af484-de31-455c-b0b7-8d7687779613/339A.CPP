/**
 * @file 339A.CPP
 * @brief Sorts the numbers in a given mathematical sum string.
 *
 * This program reads a string representing a sum (e.g., "3+2+1"), extracts
 * the numbers, sorts them in non-decreasing order, and then prints them
 * back in the same sum format.
 *
 * @algorithm
 * 1. Parse the input string character by character, converting digit
 *    characters to integers and storing them in a vector.
 * 2. Sort the vector of integers using `std::sort`.
 * 3. Print the sorted numbers, interspersed with '+' signs.
 *
 * @complexity
 * Time: O(N + K log K), where N is the length of the input string and K is the
 *       number of terms in the sum. O(N) for parsing and O(K log K) for sorting.
 * Space: O(K) to store the numbers in a vector.
 */
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

/**
 * @brief Main function to handle input, sorting, and formatted output.
 */
int main() {
  char c; // Temporary character for reading input.
  vector<int> sum; // Vector to store the integer numbers from the sum.

  // --- 1. Parsing Phase ---
  // Pre-condition: Standard input contains a string like "1+3+2".
  // Block Logic: Read the input character by character until the stream ends.
  while (cin.get(c)) {
    // '48' is the ASCII value for '0'. This check filters out non-digit
    // characters like '+' and the final newline, assuming valid input (1, 2, 3).
    // `c - 48` converts the digit character to its integer equivalent.
    if (c - 48 > 0)
      sum.push_back(c - 48);
  }

  // --- 2. Sorting Phase ---
  // Sort the vector of numbers in ascending order.
  sort(sum.begin(), sum.end(), [](int a, int b) { return a < b; });
  
  // --- 3. Printing Phase ---
  // Pre-condition: 'sum' vector is sorted.
  // Block Logic: Iterate through the sorted vector and print the numbers
  // with '+' signs in between.
  vector<int>::iterator it = sum.begin();
  // Print the first number without a preceding '+'.
  if (it != sum.end()) {
      cout << *it;
      it++;
  }
  // Print the rest of the numbers, each preceded by a '+'.
  while (it != sum.end()) {
    cout << "+" << *it;
    it++;
  }

  return 0;
}
