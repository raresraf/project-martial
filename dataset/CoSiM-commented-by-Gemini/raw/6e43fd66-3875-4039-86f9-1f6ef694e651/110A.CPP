
#include <bits/stdc++.h>
using namespace std;

/**
 * @file 110A.CPP
 * @brief Solution to Codeforces problem 110A, "Nearly Lucky Number".
 * @details The program checks if a given number is "nearly lucky". A number is
 * nearly lucky if the total count of its "lucky digits" ('4' and '7') is
 * itself a lucky number, which in this context means the count is 4 or 7.
 *
 * Algorithm:
 * The program reads the input number as a string to accommodate its large size.
 * It then iterates through the string, character by character, maintaining a
 * counter for the occurrences of '4' and '7'. After the loop finishes, it
 * checks if this final count is equal to 4 or 7. If it is, the program
 * outputs "YES"; otherwise, it outputs "NO".
 *
 * Time Complexity: O(L), where L is the number of digits in the input number,
 * as it involves a single pass through the string.
 * Space Complexity: O(L) to store the number as a string.
 */
int main() {
  // s: the input number read as a string.
  string s;
  cin >> s;
  
  // `count` stores the number of lucky digits found.
  int count = 0;

  /**
   * @brief Count the occurrences of '4' and '7'.
   */
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '4' || s[i] == '7') {
      count++;
    }
  }

  /**
   * @brief Check if the count of lucky digits is itself a lucky number (4 or 7).
   */
  if (count == 4 || count == 7) {
    cout << "YES";
  } else {
    cout << "NO";
  }

  return 0;
}
