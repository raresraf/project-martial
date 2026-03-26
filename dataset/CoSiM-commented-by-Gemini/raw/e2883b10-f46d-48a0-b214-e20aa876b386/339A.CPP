/**
 * @file 339A.CPP
 * @brief Solution to Codeforces problem 339A - Helpful Maths.
 * @description The program takes a string representing a sum of single-digit numbers
 *              (e.g., "3+1+2") and rearranges it into non-decreasing order of the numbers
 *              (e.g., "1+2+3").
 *
 * Algorithm: The core idea is to extract the numbers from the input string, sort them,
 * and then reconstruct the sum string.
 * 1. Read the input line.
 * 2. Iterate through the string, extracting all characters that are not '+' into a vector.
 *    These characters are the numbers '1', '2', and '3'.
 * 3. Sort the vector of numbers. Since the ASCII values of '1', '2', '3' are ordered,
 *    a standard sort on the character vector works correctly.
 * 4. Build a new output string by joining the sorted numbers with '+' signs.
 *
 * Time Complexity: O(N log N), dominated by the sorting step, where N is the number of
 *                  digits in the sum.
 * Space Complexity: O(N) to store the digits in a vector and to build the output string.
 */
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::vector;

int main() {
  string line, output = "";
  cin >> line;
  
  // A vector to store the character representation of the numbers.
  vector<int> numbers;

  /**
   * Block Logic: Extract all the numbers from the input string.
   * The numbers are separated by '+' signs.
   */
  for (auto chr : line) {
    if (chr != '+') {
      // Store the character code of the digit.
      numbers.push_back(chr);
    }
  }

  // Sort the numbers. The default sort works correctly on the ASCII character codes
  // ('1' < '2' < '3').
  std::sort(numbers.begin(), numbers.end());

  /**
   * Block Logic: Reconstruct the sorted sum string.
   */
  for (decltype(numbers.size()) i = 0; i != numbers.size(); ++i) {
    // Append the sorted number.
    output += numbers[i];
    // Append a '+' sign after each number except the last one.
    if (i != numbers.size() - 1) {
      output += '+';
    }
  }

  cout << output;
  return 0;
}
