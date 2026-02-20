/**
 * @file 339A.CPP
 * @brief Sorts a string of single-digit numbers separated by plus signs.
 *
 * This program reads a string representing a sum of single-digit numbers,
 * such as "3+1+2". It parses the numbers, sorts them in non-decreasing
 * order, and then prints them back in the same format.
 *
 * Algorithm:
 * 1. Read the input string.
 * 2. Extract the numeric digits from the string, ignoring the '+' signs.
 * 3. Store the digits in an integer array.
 * 4. Sort the array.
 * 5. Print the sorted digits, interleaved with '+' signs.
 *
 * Time Complexity: O(N log N), where N is the number of digits in the input string, dominated by the sort operation.
 * Space Complexity: O(N) to store the digits in an array.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  // Array to store the parsed numbers. The size 100 is a safe upper bound for typical constraints.
  int arr[100];
  string s;
  // Read the entire line containing the sum.
  cin >> s;

  int count = 0;
  /**
   * Block Logic: Parses the input string.
   * This loop iterates through the string character by character. If the character
   * is a digit, it's converted to an integer and stored in the `arr`.
   * The '+' characters are skipped.
   */
  for (int i = 0; i < s.size(); ++i) {
    if (s[i] == '+')
      continue;
    else
      // Convert char to int and store it.
      arr[count++] = s[i] - '0';
  }

  // Sort the array of numbers in non-decreasing order.
  sort(arr, arr + count);

  /**
   * Block Logic: Prints the sorted sum.
   * This loop iterates through the sorted numbers and prints them,
   * separated by '+' signs.
   */
  for (int i = 0; i < count; ++i) {
    cout << arr[i];
    // Pre-condition: Avoid printing a '+' after the last number.
    if (i == count - 1)
      break;
    cout << "+";
  }
  return 0;
}
