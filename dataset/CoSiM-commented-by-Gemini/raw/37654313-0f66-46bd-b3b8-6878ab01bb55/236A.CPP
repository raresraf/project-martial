/**
 * @file 236A.CPP
 * @brief Determines user's gender based on the parity of distinct characters in a username.
 *
 * This program reads a username as a string, counts the number of unique characters,
 * and then outputs "CHAT WITH HER!" if the count is even, and "IGNORE HIM!" if the
 * count is odd. This is a common competitive programming problem.
 *
 * Algorithm:
 * 1. Read the input string (username).
 * 2. Sort the characters of the string. This groups identical characters together.
 * 3. Iterate through the sorted string to count the number of unique characters.
 *    A character is considered unique when it differs from the one next to it.
 * 4. Check if the total count of unique characters is even or odd.
 * 5. Print the corresponding message.
 *
 * Time Complexity: O(N log N), where N is the length of the string, dominated by the sorting operation.
 * Space Complexity: O(N) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // Character array to store the username, assuming a max length of 100.
  char s[100];
  cin >> s;
  
  int i, j, c = 0;
  // Get the length of the input string.
  int l = strlen(s);
  
  /**
   * @brief Sort the string to group identical characters together.
   * This makes counting unique characters easier.
   */
  sort(s, s + l);

  /**
   * @brief Loop through the sorted string to count unique characters.
   * A new unique character is found when the current character is different
   * from the next one.
   */
  for (i = 0; i < l; i++) {
    if (s[i] != s[i + 1]) {
      c++;
    }
  }
  
  /**
   * @brief Check the parity of the unique character count.
   * If the count is even, the user is determined to be female.
   * If odd, the user is determined to be male.
   */
  if (c % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else
    cout << "IGNORE HIM!";
}
