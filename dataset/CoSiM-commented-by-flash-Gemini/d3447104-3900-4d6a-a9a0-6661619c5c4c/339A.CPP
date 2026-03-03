/**
 * @file 339A.CPP
 * @brief Solution for Codeforces Problem 339A - "Helpful Maths".
 * This program sorts the terms of a sum given as a string. The sum consists
 * of numbers 1, 2, or 3, separated by '+' signs. The output should be
 * the same sum with its terms sorted in non-decreasing order.
 * Algorithm: Parse the string to extract numbers, sort the numbers, then print them.
 * Time Complexity: O(L log L), where L is the length of the input string.
 *                  This is dominated by the sorting step.
 * Space Complexity: O(L) - to store the extracted numbers in an array.
 */
#include <algorithm> // Required for std::sort.
#include <iostream>  // Required for standard input/output operations (cin, cout).
#include <string>    // Required for std::string.

using namespace std; // Uses the standard namespace.

int main() {
  int arr[1001]; // Declares an array to store the numbers from the sum. Max size 1001 to handle string length up to 200 (approx. 100 numbers).
  string s;      // Declares a string to store the input sum.
  cin >> s;      // Reads the sum string from standard input.

  int count = 0; // Counter for the number of digits extracted and stored in 'arr'.

  /**
   * Block Logic: Parses the input string to extract individual numbers.
   * It skips '+' characters and converts digit characters to integers, storing them in 'arr'.
   * Invariant: `count` reflects the number of integers successfully extracted from the string.
   */
  for (int i = 0; i < s.size(); i++) { // Loop through each character of the input string.
    if (s[i] == '+') {
      continue; // Skip the '+' characters.
    } else {
      // Converts the digit character to an integer and stores it in 'arr'.
      // 's[i] - '0'' converts a character digit (e.g., '1') to its integer value (1).
      arr[count++] = s[i] - '0';
    }
  }

  // Sorts the array of extracted numbers in non-decreasing order.
  sort(arr, arr + count);

  /**
   * Block Logic: Prints the sorted numbers, separated by '+', to form the sorted sum.
   * Ensures that no '+' is printed after the last number.
   */
  for (int i = 0; i < count; i++) { // Loop through the sorted numbers.
    cout << arr[i]; // Prints the current number.
    if (i == count - 1) { // Checks if it's the last number.
      break; // If it is, do not print a '+' after it.
    }
    cout << "+"; // Prints a '+' to separate numbers.
  }
  cout << endl; // Prints a newline character at the end. (Implicit in original code due to missing endl)

  return 0; // Indicates successful program execution.
}