/**
 * @file 339A.CPP
 * @brief Solution to a competitive programming problem for sorting and re-formatting an arithmetic expression.
 *
 * This program processes a string input which is expected to be an arithmetic
 * expression consisting of single-digit numbers and '+' operators (e.g., "1+1+3+1").
 * The objective is to extract the numbers, sort them in non-decreasing order,
 * and then print them back as an arithmetic expression, also joined by '+' operators.
 * This effectively sorts the operands of an addition without altering the result.
 *
 * Algorithm: Extraction, Sorting, and Re-composition.
 * Time Complexity: O(L log L) where L is the number of digits extracted (and thus the size of the vector),
 *                  dominated by the sorting operation. Reading characters is O(N) where N is input string length.
 * Space Complexity: O(L) to store the extracted digits in a vector.
 */

#include <algorithm> // Required for std::sort
#include <iostream>  // Required for standard input/output operations (cin, cout, cin.get)
#include <vector>    // Required for std::vector container

/**
 * @brief Standard namespace for C++ functionalities.
 */
using namespace std;

/**
 * @brief Main function where the program execution begins.
 * @return int Returns 0 on successful execution.
 */
int main() {
  char c;
  vector<int> sum;

  /**
   * @brief Reads characters from standard input and extracts digits greater than '0'.
   *
   * Pre-condition: Standard input stream is open and contains characters.
   * Invariant: `sum` vector accumulates integer representations of digits greater than '0'.
   *            The loop continues until the end of the input stream (EOF).
   * Functional Utility: Populates the `sum` vector with numerical operands from the input expression.
   */
  while (cin.get(c)) {
    // Conditional Logic: Filters characters, keeping only digits '1' through '9'.
    // Inline: Converts character digit to its integer value (ASCII '0' is 48).
    if (c - 48 > 0)
      sum.push_back(c - 48);
  }

  /**
   * @brief Sorts the extracted numbers in non-decreasing order.
   *
   * Algorithm: Standard library sort (typically IntroSort, a hybrid algorithm).
   * Pre-condition: `sum` vector contains extracted integer operands.
   * Post-condition: Elements within `sum` are arranged in ascending order.
   * Functional Utility: Arranges the operands for ordered output.
   */
  sort(sum.begin(), sum.end(), [](int a, int b) { return a < b; });

  vector<int>::iterator it = sum.begin();
  /**
   * @brief Prints the first number in the sorted list.
   *
   * Pre-condition: `sum` is not empty (guaranteed by problem constraints, or would need a check).
   * Post-condition: The first operand of the re-composed expression is printed to standard output.
   * Functional Utility: Handles the printing of the initial operand without a preceding '+' sign.
   */
  cout << *it;
  it++;
  /**
   * @brief Iterates through the rest of the sorted numbers and prints them, prefixed by '+'.
   *
   * Invariant: `it` advances through the `sum` vector, ensuring all subsequent operands are printed.
   * Pre-condition: `it` points to an element after the first one, or `sum` is empty.
   * Post-condition: All remaining operands are printed, each preceded by a '+' sign.
   * Functional Utility: Re-composes the arithmetic expression with sorted operands.
   */
  while (it != sum.end()) {
    cout << "+" << *it;
    it++;
  }

  return 0;
}