/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem.
 * @details The program takes a string representing a sum of single-digit numbers
 * (e.g., "3+2+1") and rearranges the numbers into non-decreasing order
 * (e.g., "1+2+3").
 *
 * @algorithm
 * 1. Read the input string containing the sum.
 * 2. Parse the string to extract the digit characters, ignoring the '+' symbols.
 * 3. Store the digits in a character vector.
 * 4. Sort the vector of digits in non-decreasing order.
 * 5. Reconstruct the output string by printing the sorted digits, separated by '+'.
 *
 * @complexity
 * Time Complexity: O(N log N), where N is the number of digits in the sum. The
 * complexity is dominated by the sorting operation.
 * Space Complexity: O(N) to store the digits in a vector.
 */
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

// Using the standard namespace to avoid prefixing library calls with `std::`.
using namespace std;

/**
 * @brief A custom comparator for sorting characters.
 * @details Despite its name `desc`, this function implements ascending order
 * sorting by returning true if `i` is less than `j`.
 * @param i The first character to compare.
 * @param j The second character to compare.
 * @return True if `i` should come before `j`, false otherwise.
 */
bool desc(char i, char j) { return i < j; }

int main() {
  // `kata` (Indonesian/Malay for "word") stores the input string sum.
  string kata;
  cin >> kata;

  // `operands` will store the digit characters extracted from the string.
  vector<char> operands;
  /**
   * @block
   * @brief Parse the input string to extract digits.
   * @invariant Iterates through the entire input string `kata`.
   */
  for (int i = 0; i < kata.length(); i++) {
    if (kata[i] != '+') {
      operands.push_back(kata[i]);
    }
  }

  // Sort the extracted digits. The `desc` function provides ascending order.
  sort(operands.begin(), operands.end(), desc);

  /**
   * @block
   * @brief Reconstruct and print the sorted sum string.
   * @invariant Iterates through the sorted `operands` vector.
   */
  for (int i = 0; i < operands.size(); i++) {
    // A '+' is printed before every operand except the very first one.
    if (i != 0) {
      cout << "+";
    }
    cout << operands[i];
  }
}