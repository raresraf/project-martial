/**
 * @file raw/45df1ab3-231c-4eab-aae3-efb483369066/339A.CPP
 * @brief Solution to Codeforces problem 339A - Helpful Maths.
 *
 * @algorithm
 * The problem requires rearranging a given sum of numbers (1, 2, and 3) into
 * non-decreasing order. For example, "3+2+1" should become "1+2+3".
 *
 * The core idea is to extract the numbers from the input string, sort them,
 * and then reconstruct the string with the sorted numbers.
 *
 * The algorithm implemented is as follows:
 * 1. Read the entire sum as a single string (e.g., "3+2+1").
 * 2. Create a character array to store only the digits from the string.
 * 3. Iterate through the input string at every second position (0, 2, 4, ...)
 *    to pick out the digit characters, skipping the '+' signs.
 * 4. Store these digit characters in the new array.
 * 5. Sort the character array. Since the ASCII values of '1', '2', and '3' are
 *    in the correct order, sorting the characters is equivalent to sorting the
 *    numbers they represent.
 * 6. Iterate through the sorted array of digits.
 * 7. Print each digit, followed by a '+' sign, until the last digit.
 * 8. Print the last digit followed by a newline.
 *
 * @complexity
 * - Time Complexity: O(L log L), where L is the number of digits in the sum.
 *   This is dominated by the `std::sort` operation.
 * - Space Complexity: O(L), to store the extracted digits in a separate array.
 */
#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  string s; // To store the input sum string.
  cin >> s;
  
  // A C-style character array to hold the extracted digits.
  char array[200];
  int count = 0; // To count how many digits are extracted.

  /**
   * @brief Extract the digits from the input string.
   * The numbers are at even indices (0, 2, 4, ...).
   */
  for (int i = 0; i <= s.size() / 2; i++) {
    array[i] = s[2 * i];
    count++;
  }

  // Sort the array of digit characters.
  sort(array, array + count);

  /**
   * @brief Print the sorted digits, separated by '+'.
   */
  for (int i = 0; i < count; i++) {
    if (i == count - 1)
      // Print the last digit followed by a newline.
      cout << array[i] << endl;
    else
      // Print the current digit followed by a '+'.
      cout << array[i] << "+";
  }
  return 0;
}
