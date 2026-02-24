/**
 * @file raw/461b85e5-bf3d-4a83-9ed0-7e4f2f392b27/110A.CPP
 * @brief Solution to Codeforces problem 110A - Nearly Lucky Number.
 *
 * @algorithm
 * The problem defines a "nearly lucky number" based on two conditions:
 * 1. A "lucky digit" is either '4' or '7'.
 * 2. A "lucky number" is a positive integer consisting only of lucky digits (e.g., 4, 7, 47, 774).
 * 3. A number is "nearly lucky" if the total count of lucky digits within it is
 *    itself a lucky number. Since the input number can have up to 18 digits,
 *    the count of lucky digits can be at most 18. The only lucky numbers less than
 *    or equal to 18 are 4 and 7.
 *
 * Therefore, a number is nearly lucky if it contains exactly 4 or exactly 7
 * lucky digits ('4' or '7').
 *
 * The algorithm is as follows:
 * 1. Read the input number as a string to easily access its digits.
 * 2. Initialize a counter to zero.
 * 3. Iterate through each character of the string.
 * 4. If a character is '4' or '7', increment the counter.
 * 5. After the loop, check if the final count is equal to 4 or 7.
 * 6. If it is, print "YES". Otherwise, print "NO".
 *
 * @complexity
 * - Time Complexity: O(L), where L is the number of digits in the input number,
 *   as the code iterates through the string once.
 * - Space Complexity: O(L), to store the input number as a string.
 */
#include <bits/stdc++.h> // Includes most standard libraries, including iostream and cstring.
using namespace std;
int main() {
  // `cou` will count the number of lucky digits ('4' or '7').
  int cou = 0;
  // `s` is a C-style string to store the input number.
  char s[100];
  cin >> s;

  /**
   * @brief Iterate through the digits of the number to count lucky ones.
   */
  for (int i = 0; i < strlen(s); i++) {
    // If the current digit is a lucky digit...
    if (s[i] == '4' || s[i] == '7')
      // ...increment the counter.
      cou++;
  }

  /**
   * @brief Check if the count of lucky digits is itself a lucky number (4 or 7).
   */
  if (cou == 4 || cou == 7) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  
  // The original code is missing a return statement.
  // In C++, main() is special and will implicitly return 0 if control reaches the end.
}
