/**
 * @b58808ce-dbb5-4fd9-adc4-35a0c39d40fc/339A.CPP
 * @brief Solution for a competitive programming problem involving parsing and sorting digits from a sum string.
 *
 * This program reads a string representing a sum (e.g., "1+5+2"), extracts the individual digits,
 * counts the occurrences of digits 1, 2, and 3, and then reconstructs and prints the sum
 * with the digits sorted in non-decreasing order, re-interspersed with '+' signs.
 * This approach uses counting sort principles for efficiency when the range of digits is small.
 *
 * Algorithm:
 * 1. Read the input string.
 * 2. Initialize an array to count occurrences of digits 1, 2, and 3.
 * 3. Iterate through the input string, extracting digits (characters at even indices).
 * 4. Increment the count for each extracted digit in the counting array.
 * 5. Keep track of the total number of digits (which is also the number of plus signs needed).
 * 6. Iterate from digit 1 to 3. For each digit, print it as many times as its count,
 *    appending a '+' sign if there are more digits to print.
 *
 * Time Complexity: O(N) where N is the length of the input string, as it involves a single pass
 *                  to count digits and another pass over a fixed-size array (digits 1-3).
 * Space Complexity: O(1) as the counting array size is fixed (4 elements).
 */

#include <bits/stdc++.h> // Functional Utility: Includes most standard C++ libraries, common in competitive programming.
using namespace std;

int main() {
  string inp;
  // Functional Utility: Reads the entire input string (e.g., "1+2+3").
  cin >> inp;
  int a[4] = {0}; // Functional Utility: Array to store counts of digits 1, 2, and 3. Index 0 is unused.
  int len = inp.length();
  // Functional Utility: Counter for the number of digits extracted, also used to determine '+' placement.
  int plus_count = 0;
  // Block Logic: Iterates through the input string to extract digits and count their occurrences.
  // Pre-condition: 'inp' contains digits at even indices and '+' at odd indices.
  // Invariant: 'i' increments by 2 to skip '+' characters; 'a[j]' stores the count of digit 'j'.
  for (int i = 0; i < len; i += 2) {
    int j = (int)inp[i] - 48; // Inline: Converts character digit to its integer value. ASCII '0' is 48.
    a[j]++; // Functional Utility: Increments the count for the extracted digit.
    plus_count++; // Functional Utility: Tracks the total number of digits.
  }

  // Block Logic: Reconstructs and prints the sorted sum string.
  // Pre-condition: 'a' contains counts of digits, 'plus_count' is total digits.
  // Invariant: Digits are printed in increasing order (1, then 2, then 3), separated by '+'.
  for (int i = 1; i < 4; i++) { // Functional Utility: Iterates through possible digits (1, 2, 3).
    // Block Logic: Prints digit 'i' as many times as it appeared in the input.
    // Pre-condition: 'a[i]' is the count of digit 'i'.
    // Invariant: Digit 'i' is printed, and 'plus_count' is decremented until 'a[i]' is zero.
    while (a[i] > 0) {
      cout << i; // Functional Utility: Prints the current digit.
      plus_count--; // Functional Utility: Decrements the counter for remaining digits to print.
      // Conditional Logic: Prints a '+' sign if there are more digits to follow.
      if (plus_count > 0)
        cout << '+';
      a[i]--; // Functional Utility: Decrements the count for the current digit.
    }
  }
  // No return 0; explicitly, but program execution ends after printing.
  // In competitive programming, omitting return 0; is common for main if it reaches the end.
}
