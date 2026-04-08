/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem from Codeforces (Problem 339A).
 * @details The program takes a string representing a sum of single digits (e.g., "3+2+1")
 * and rearranges it into a non-decreasing sum (e.g., "1+2+3").
 *
 * @algorithm
 * 1. Read the input string containing the sum.
 * 2. Create a vector of characters to store the digits.
 * 3. Iterate through the input string, picking out the characters at even indices
 *    (0, 2, 4, ...), which correspond to the digits, and add them to the vector.
 * 4. Sort the vector of digit characters. Since they are single digits, character
 *    sorting is equivalent to numerical sorting.
 * 5. Iterate through the sorted vector.
 * 6. Print each digit, followed by a "+" sign, ensuring no "+" is printed after
 *    the last digit.
 *
 * @time_complexity O(N log N), where N is the number of digits in the sum. This is
 * dominated by the sorting step.
 * @space_complexity O(N) to store the digits in a vector.
 */
#include <algorithm>
#include <ctype.h>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

int main(void) {
  string test; // Input string, e.g., "3+1+2"
  cin >> test;
  vector<char> vec; // Vector to hold the digit characters

  /**
   * @block_logic Extract the digits from the input string.
   * The loop increments by 2 to skip the '+' signs.
   */
  for (int i = 0; i < test.length(); i += 2) {
    vec.push_back(test[i]);
  }

  // Sort the vector of digits.
  sort(vec.begin(), vec.end());

  /**
   * @block_logic Reconstruct and print the sorted sum string.
   */
  for (int i = 0; i < vec.size(); i++) {
    cout << vec[i];

    // Add a '+' sign after each digit except the last one.
    if (i != (vec.size() - 1)) {
      cout << "+";
    }
  }
  return 0;
}