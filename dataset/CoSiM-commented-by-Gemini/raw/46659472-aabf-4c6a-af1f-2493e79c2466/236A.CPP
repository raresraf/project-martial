/**
 * @file 236A.CPP
 * @brief Solution for a programming contest problem to determine gender based on username.
 * @details The problem states that if the number of distinct characters in a user's
 * name is even, the user is female ("CHAT WITH HER!"), and if it's odd, the user is male
 * ("IGNORE HIM!").
 *
 * Algorithm:
 * 1. Read the username as a string.
 * 2. Count the number of distinct characters in the string. The implementation does this
 *    by first sorting the string, which groups identical characters together. Then, it
 *    iterates through the sorted string and increments a counter whenever a character
 *    is different from the next one.
 * 3. Check if the count of distinct characters is even or odd.
 * 4. Print "CHAT WITH HER!" for an even count and "IGNORE HIM!" for an odd count.
 *
 * Note on Implementation: The loop for counting distinct characters has a potential
 * out-of-bounds access at the last character (`s[i+1]`). While this might work in some
 * environments, a more robust method would be to use `std::set` or the `std::unique` algorithm.
 *
 * Time Complexity: O(L log L), where L is the length of the string, dominated by the sorting step.
 * Space Complexity: O(L) to store the string, although sorting happens in-place.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  int x = 0;
  cin >> s;
  
  // Sort the string to group identical characters together.
  sort(s.begin(), s.end());
  
  /**
   * @block
   * @description Counts the number of distinct characters in the sorted string.
   * It increments the count `x` each time a character differs from the one
   * immediately following it. This works because all identical characters are now adjacent.
   */
  for (int i = 0; i < s.size(); i++) {
    if (s[i] != s[i + 1]) {
      x++;
    }
  }
  
  /**
   * @block
   * @description Determines the output based on the parity of the distinct character count.
   */
  if (x % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
}