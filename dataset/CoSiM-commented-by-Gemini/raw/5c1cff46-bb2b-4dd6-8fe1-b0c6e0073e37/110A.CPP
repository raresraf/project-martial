/**
 * @file 110A.CPP
 * @brief Solution for the "Nearly Lucky Number" problem using string manipulation.
 * @details This program determines if a number is "nearly lucky". A number is nearly
 * lucky if the count of its lucky digits (4 and 7) is itself a lucky number (4 or 7).
 * This implementation reads the number as a string to simplify digit access.
 *
 * @algorithm String Iteration and Counting
 * 1. Read the input number as a `std::string`.
 * 2. Initialize a counter for lucky digits to zero.
 * 3. Iterate through each character of the string.
 * 4. If a character is '4' or '7', increment the counter.
 * 5. After iterating through the string, check if the final counter value is 4 or 7.
 * 6. Print "YES" if it is, and "NO" otherwise.
 *
 * @complexity
 * - Time: O(L), where L is the number of digits in the input number (the length of the string).
 * - Space: O(L), to store the input number as a string.
 */

#include <bits/stdc++.h> // Includes all standard C++ libraries.
using namespace std;

int main(void) {
  // s: stores the input number as a string.
  string s;
  cin >> s;

  // ans: counter for the number of lucky digits found.
  int ans = 0;

  /**
   * @brief Iterate through the string to count lucky digits.
   * This loop processes each character of the input string.
   */
  // Invariant: `ans` holds the count of lucky digits in the substring s[0...i-1].
  for (int i = 0; i < s.length(); ++i) {
    // Pre-condition: Check if the current character is a lucky digit.
    if (s[i] == '7' || s[i] == '4') {
      ans++;
    }
  }

  /**
   * @brief Check if the count of lucky digits is itself a lucky number.
   * A ternary operator is used for a concise conditional output.
   */
  cout << ((ans == 4 || ans == 7) ? "YES" : "NO") << '\n';

  return 0; // Indicate successful execution.
}
