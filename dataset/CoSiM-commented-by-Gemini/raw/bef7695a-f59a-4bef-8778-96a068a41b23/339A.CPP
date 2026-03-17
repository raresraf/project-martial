/**
 * @file 339A.CPP
 * @brief Solution to a competitive programming problem, likely Codeforces 339A "Helpful Maths".
 *
 * @details
 * This program rearranges a string representing a sum of single digits (e.g., "3+2+1")
 * so that the numbers appear in non-decreasing order (e.g., "1+2+3").
 *
 * @algorithm
 * This solution uses an in-place sorting algorithm (similar to Bubble Sort or
 * Selection Sort) that specifically targets the digits in the string.
 * 1. The input string is read.
 * 2. A nested loop iterates through the string, only considering the characters
 *    at even indices (0, 2, 4, ...), which correspond to the digits.
 * 3. It compares pairs of digits and swaps them if they are in the wrong order.
 * 4. The '+' signs at odd indices are never touched or compared.
 * 5. After the loops complete, the digits in the string are sorted, and the
 *    modified string is printed.
 *
 * @complexity
 * - Time Complexity: O(N^2), where N is the length of the string, due to the
 *   nested loops for sorting.
 * - Space Complexity: O(N) to store the input string, with O(1) auxiliary space
 *   as the sorting is performed in-place.
 */

#include <iostream>
#include <string> // Included for swap, though <algorithm> is more standard
#include <algorithm> // For std::swap

using namespace std;
string s;

/**
 * @brief Main function and entry point of the program.
 */
int main() {
  cin >> s;

  /**
   * @brief In-place sort of the digits in the string.
   * The outer loop selects a digit, and the inner loop compares it with all
   * subsequent digits. The loops increment by 2 to step over the '+' signs.
   */
  for (int i = 0; i < s.size(); i += 2)
    for (int j = i + 2; j < s.size(); j += 2)
      // If a preceding digit is greater than a following one, swap them.
      if (s[i] > s[j])
        swap(s[i], s[j]);
  
  // Print the string after the digits have been sorted in-place.
  cout << s;
}
