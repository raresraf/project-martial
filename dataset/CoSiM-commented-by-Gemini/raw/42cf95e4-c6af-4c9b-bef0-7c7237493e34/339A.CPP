/**
 * @file 339A.CPP
 * @brief Solution to Codeforces problem 339A - Helpful Maths.
 *
 * @algorithm
 * The problem requires sorting the numbers in a string that represents a sum
 * (e.g., "3+1+2" should become "1+2+3").
 *
 * This solution uses a standard and clean approach:
 * 1. Extract the numbers from the input string.
 * 2. Store these numbers in a `std::vector<int>`.
 * 3. Use the `std::sort` algorithm to sort the vector.
 * 4. Reconstruct the output string by iterating through the sorted vector and
 *    printing the numbers separated by '+'.
 *
 * @complexity
 * - Time Complexity: O(N log N), where N is the number of summands. This is
 *   dominated by the `std::sort` operation. The string parsing and printing
 *   are linear, O(L), where L is the string length.
 * - Space Complexity: O(N), to store the numbers in the vector.
 */
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

/**
 * @brief Converts a digit character to its integer value.
 * @param a The character to convert (e.g., '1').
 * @return The corresponding integer (e.g., 1).
 */
int conver(char a) { return a - '0'; }

int main() {
  string a;
  vector<int> num_char2;
  cin >> a;
  /**
   * @brief Extracts the numbers from the input string and stores them in a vector.
   * It iterates with a step of 2 to skip the '+' characters.
   */
  for (int i = 0; i < a.length(); i = i + 2) {
    num_char2.push_back(conver(a[i]));
  }
  // Sort the vector of numbers in non-decreasing order.
  sort(num_char2.begin(), num_char2.end());
  /**
   * @brief Prints the sorted numbers, re-joined by '+' signs.
   * A check is made to ensure no '+' is printed after the last number.
   */
  for (vector<int>::iterator i = num_char2.begin(); i != num_char2.end(); i++) {
    if (i == num_char2.end() - 1) {
      cout << *i;
    } else {
      cout << *i << "+";
    }
  }
}