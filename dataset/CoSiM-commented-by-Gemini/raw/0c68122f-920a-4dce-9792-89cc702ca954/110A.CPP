/**
 * @file 110A.CPP
 * @brief Solves the "Nearly Lucky Number" problem with a generalized check.
 * @details A number is "nearly lucky" if the count of its lucky digits ('4' and '7') is itself a lucky number.
 * This implementation interprets "a lucky number" as any number composed entirely of the digits '4' and '7'.
 *
 * @algorithm
 * 1. Read the input number as a string `s`.
 * 2. Count the occurrences of '4' and '7' in `s`. Let this count be `c`.
 * 3. Convert the integer count `c` into a new string `ct`.
 * 4. Iterate through the characters of the count-string `ct`.
 * 5. If any character in `ct` is not '4' or '7', the count is not a lucky number.
 * 6. If all characters in `ct` are either '4' or '7', the count is a lucky number.
 * 7. Print "YES" if the count is a lucky number, otherwise "NO".
 *
 * @note For the specific constraints of this problem (input number length <= 18), the maximum count of
 * lucky digits is 18. The only lucky numbers (composed of 4s and 7s) in this range are 4 and 7.
 * Therefore, this generalized check gives the same result as a simpler `if (c == 4 || c == 7)`.
 *
 * @complexity
 * Time complexity: O(L), where L is the number of digits in the input number.
 * Space complexity: O(L) to store the input string and the count string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // i: loop counter. c: used first as a counter, then as a boolean flag.
  int i = 0, c = 0;
  // s: the input number string. ct: the string representation of the lucky digit count.
  string s, ct;
  cin >> s;
  
  /**
   * @brief First pass: count the lucky digits in the input number.
   * @invariant Iterates through the string `s`.
   */
  while (s[i] != '\0') {
    if (s[i] == '4' || s[i] == '7') {
      c++;
    }
    i++;
  }
  
  // Convert the integer count to a string for the second pass.
  ct = to_string(c);
  
  // Reset variables for the second pass. `c` is reused as a flag.
  i = 0;
  c = 0;
  
  /**
   * @brief Second pass: check if the count itself is a lucky number.
   * @invariant Iterates through the count string `ct`.
   */
  while (ct[i] != '\0') {
    // If a digit in the count is not '4' or '7', it's not a lucky number.
    if (ct[i] != '4' && ct[i] != '7') {
      c = 1; // Set flag to indicate a non-lucky digit was found.
      break;
    }
    i++;
  }
  
  // If the flag `c` is 1, the number is not nearly lucky.
  if (c == 1)
    cout << "NO";
  else // Otherwise, it is.
    cout << "YES";
}