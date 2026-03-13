/**
 * @file 339A.CPP
 * @brief Solution for CodeForces Problem 339A - "Helpful Maths".
 *
 * @details
 * The program takes a string representing a sum of the numbers 1, 2, and 3
 * (e.g., "3+2+1") and rearranges the numbers in non-decreasing order
 * (e.g., "1+2+3").
 *
 * @algorithm Multi-pass Bucket Sort
 * The code implements a stable sorting logic by making three passes over the input string.
 * 1. First Pass: It iterates through the input string, and for every '1' it finds,
 *    it appends "1+" to the result string.
 * 2. Second Pass: It iterates through the input string again, and for every '2' it finds,
 *    it appends "2+" to the result string.
 * 3. Third Pass: It does the same for '3'.
 *
 * After these three passes, the `result` string will contain the numbers in sorted order,
 * each followed by a '+'.
 *
 * Finally, it removes the trailing '+' from the end of the `result` string before printing.
 *
 * @time_complexity O(L), where L is the length of the string. The algorithm makes a fixed
 *                  number of passes (3) over the input string.
 * @space_complexity O(L) to store the result string.
 */
#include <iostream>
#include <string>
#include <vector> // Not strictly needed but good practice for C++

using namespace std;

int main() {
  string expression, result;
  cin >> expression;
  int length = expression.length();

  /**
   * @brief First pass: append all '1's to the result.
   * Iterates through the expression, stepping by 2 to only inspect the digits.
   */
  for (int i = 0; i < length; i += 2) {
    if (expression[i] == '1')
      result.append("1+");
  }

  /**
   * @brief Second pass: append all '2's to the result.
   */
  for (int i = 0; i < length; i += 2) {
    if (expression[i] == '2')
      result.append("2+");
  }

  /**
   * @brief Third pass: append all '3's to the result.
   */
  for (int i = 0; i < length; i += 2) {
    if (expression[i] == '3')
      result.append("3+");
  }

  // After all passes, the result string has a trailing '+'. Remove it.
  if (!result.empty()) {
    result.pop_back();
  }
  
  cout << result;

  return 0;
}
