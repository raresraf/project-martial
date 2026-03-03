
#include <bits/stdc++.h>
using namespace std;

/**
 * @file 339A.CPP
 * @brief Solution to Codeforces problem 339A, "Helpful Maths".
 * @details The program takes a string representing a sum of single-digit numbers
 * and rearranges it into non-decreasing order.
 *
 * Algorithm:
 * This implementation follows a straightforward "extract, sort, and print"
 * approach.
 * 1.  It iterates through the input string and extracts all the digit
 *     characters, converting them to integers and storing them in an array.
 *     The '+' characters are ignored.
 * 2.  It sorts the integer array in non-decreasing order.
 * 3.  It iterates through the sorted array, printing each number followed by a
 *     '+' separator, omitting the separator after the last number.
 *
 * Time Complexity: O(L log L), where L is the number of digits in the sum.
 * The complexity is dominated by the sort operation.
 * Space Complexity: O(L) to store the array of integers.
 */
int main() {
  string s;
  cin >> s;

  int array[100]; // Array to hold the integer digits.
  int c = 0;      // Counter for the number of digits.

  /**
   * @brief Extract digit characters from the string and convert to integers.
   */
  for (int i = 0; i < s.size(); i++) {
    if (s[i] != '+')
      // Convert char to int ('1' - 48 = 1) and store in the array.
      array[c++] = s[i] - 48;
  }

  // Sort the array of numbers.
  sort(array, array + c);

  /**
   * @brief Print the sorted numbers with '+' separators.
   */
  for (int i = 0; i < c; i++) {
    cout << array[i];
    // Stop after printing the last number to avoid a trailing '+'.
    if (i == c - 1)
      break;
    else
      cout << "+";
  }

  return 0;
}
