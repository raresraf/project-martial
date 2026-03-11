/**
 * @file 110A.CPP
 * @brief Solution to the "Nearly Lucky Number" problem.
 *
 * This program determines if a given number (represented as a string) is a "nearly lucky number".
 * A number is nearly lucky if the count of "lucky digits" (4s and 7s) within it is itself a lucky number.
 * Algorithm: Count occurrences of '4' and '7' in the input string. Then, check if this count is equal to 4 or 7.
 * Time Complexity: O(L) where L is the length of the input number string, due to string iteration.
 * Space Complexity: O(L) to store the input number string.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to determine if an input number is a "nearly lucky number".
 *
 * Reads a number as a string, counts the lucky digits ('4' and '7'), and based on this count,
 * determines if the original number is "nearly lucky".
 * @return 0 on successful execution.
 */
int main() {
  int count = 0; // Invariant: 'count' stores the number of lucky digits encountered so far.
  char s[21];    // Declare a character array to store the input number as a string. Max length 20 + null terminator.

  // Block Logic: Read the input number string.
  // Pre-condition: User provides a string of digits (max 20 characters).
  // Invariant: 's' contains the input number string.
  scanf("%s", s);

  // Block Logic: Iterate through the input string to count occurrences of lucky digits ('4' and '7').
  // Pre-condition: 's' is a null-terminated string.
  // Invariant: 'count' accurately reflects the number of '4's and '7's in 's' up to the current index.
  for (int i = 0; i < strlen(s); i++)
    if (s[i] == '4' || s[i] == '7')
      count++;

  // Block Logic: Determine if the count of lucky digits is itself a lucky number (4 or 7).
  // Uses a switch statement for clarity to check against these specific lucky numbers.
  // Pre-condition: 'count' holds the total number of lucky digits.
  // Post-condition: "YES" is printed if 'count' is 4 or 7, otherwise "NO" is printed.
  switch (count) {
  case 4:
  case 7:
    printf("YES\n");
    break;
  default:
    printf("NO\n");
  }
  return 0;
}
