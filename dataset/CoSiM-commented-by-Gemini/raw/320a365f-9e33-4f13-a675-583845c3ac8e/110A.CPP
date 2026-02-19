/**
 * @file 110A.CPP
 * @brief Solves the "Nearly Lucky Number" problem.
 * @details A number is considered "nearly lucky" if the number of "lucky digits"
 * (which are '4' and '7') in it is itself a lucky number. This program checks if an
 * input number satisfies this condition.
 *
 * @algorithm The program reads the input number as a string to easily iterate over
 * its digits. It then performs a single pass through the string, counting every
 * occurrence of the characters '4' and '7'. Finally, it checks if the resulting
 * count is equal to 4 or 7. If it is, the number is declared "nearly lucky";
 * otherwise, it is not.
 *
 * @time_complexity O(L), where L is the number of digits in the input number, as the
 * program iterates through the string once.
 * @space_complexity O(L), to store the input number as a string.
 */
// author @Nishant

#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  cin >> s;
  // `count` will store the number of lucky digits ('4' and '7') found.
  int count = 0;
  /**
   * Block Logic: Iterate through each character of the input string to count lucky digits.
   */
  for (int i = 0; i < s.size(); i++) {
    // Inline: This is an obscure way to check for '4' or '7'.
    // It relies on ASCII value arithmetic: '4' - '1' (49) is 3, and '7' - '1' (49) is 6.
    // A clearer check would be `s[i] == '4' || s[i] == '7'`.
    if (char(s[i]) - 49 == 3 || char(s[i]) - 49 == 6) {
      count++;
    }
  }
  /**
   * Block Logic: Determine if the number is "nearly lucky".
   * This is true if the count of lucky digits is itself a lucky number (4 or 7).
   */
  if (count == 4 || count == 7) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}