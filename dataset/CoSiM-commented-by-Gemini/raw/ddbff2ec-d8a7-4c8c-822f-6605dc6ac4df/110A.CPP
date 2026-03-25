#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

/**
 * @file 110A.CPP
 * @brief Solution to a competitive programming problem, "Nearly Lucky Number".
 *
 * Problem Statement:
 * A "lucky number" is a positive integer whose decimal representation contains
 * only the digits 4 and 7. For example, 47, 744, 4 are lucky numbers.
 * A number is called "nearly lucky" if the count of lucky digits ('4' or '7')
 * in it is a lucky number.
 *
 * This program determines if a given number is a nearly lucky number.
 *
 * Algorithm:
 * This implementation reads the input number as a string to easily iterate
 * over its digits.
 *
 * 1. Read the input number as a string `s`.
 * 2. Initialize a counter `cnt` to zero.
 * 3. Iterate through each character of the string `s`.
 * 4. For each character, if it is '4' or '7', increment the counter `cnt`.
 * 5. After the loop, check if the final count `cnt` is itself a lucky number.
 *    For the given constraints (input number up to 10^18), the maximum
 *    possible count is 18. The only single-digit or two-digit lucky numbers
 *    are 4 and 7. Therefore, we only need to check if the count is 4 or 7.
 * 6. Print "YES" if the count is 4 or 7, otherwise print "NO".
 *
 * Time Complexity: O(L), where L is the number of digits in the input number.
 * Space Complexity: O(L) to store the input string.
 */

int main() {
  string s;
  // Read the number as a string, which simplifies digit-by-digit access.
  cin >> s;
  ll len = s.length();
  // Counter for the number of lucky digits ('4' or '7').
  ll cnt = 0;

  // Block Logic: Iterate through the string to count lucky digits.
  // Invariant: 'cnt' holds the number of lucky digits in the first 'i' characters.
  for (int i = 0; i < len; i++) {
    // Pre-condition: Check if the current character is a lucky digit.
    if (s[i] == '7' || s[i] == '4') {
      cnt++;
    }
  }

  // Final check: Determine if the count of lucky digits is itself a lucky number.
  if (cnt == 4 || cnt == 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
}