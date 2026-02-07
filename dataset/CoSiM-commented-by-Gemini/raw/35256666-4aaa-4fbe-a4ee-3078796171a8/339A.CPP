/**
 * @file 339A.CPP
 * @brief Solution to a competitive programming problem involving sorting numbers in a string expression.
 * This program reads a string representing a sum of numbers (e.g., "1+3+2"),
 * sorts the numeric operands, and then prints the expression with the sorted
 * numbers separated by '+' signs.
 *
 * Algorithm: String parsing, sorting, and reconstruction.
 * Time Complexity: O(L log L) where L is the length of the input string due to sorting.
 * Space Complexity: O(L) for storing the string.
 */
#include <algorithm>
#include <iostream>
#include <string.h>

using namespace std;

/**
 * @brief Main function of the program.
 * Reads a string `conta` representing an arithmetic expression, sorts its numeric
 * components, and then prints the reordered expression with '+' signs.
 * @return `0` upon successful execution.
 */
int main() {
  string conta{};
  cin >> conta;
  /**
   * @brief Sorts the characters in the input string.
   * Functional Utility: Due to ASCII values, '+' characters will move to the beginning,
   * followed by the sorted numeric characters. This effectively sorts the numbers
   * in the expression while grouping the '+' signs.
   */
  sort(conta.begin(),
       conta.end()); // all '+' will be in first place and the numbers sorted.
  /**
   * @brief Loop to print sorted numbers separated by '+'.
   * Invariant: Iterates through the sorted string, printing each non-'+' character
   * followed by a '+' sign, effectively reconstructing the sorted expression.
   * The last character is handled outside the loop to avoid a trailing '+'.
   */
  for (int i = 0; i < conta.length() - 1; i++) {
    if (conta[i] != '+') {
      cout << conta[i] << '+';
    }
  }
  cout << conta[conta.length() - 1]; // Prints the last character without a trailing '+'

  return 0;
}