/**
 * @file 236A.CPP
 * @brief Solves the "Boy or Girl" problem.
 *
 * @details
 * The program determines the gender of a user based on the number of distinct
 * characters in their username. If the number of distinct characters is even,
 * the user is female ("CHAT WITH HER!"). If it's odd, the user is male ("IGNORE HIM!").
 *
 * Algorithm:
 * 1. Read the username string.
 * 2. Sort the string alphabetically. This groups all identical characters together.
 * 3. Count the number of times a character is different from the one preceding it.
 *    This count (`ans`) will be one less than the total number of distinct characters.
 * 4. The number of distinct characters is therefore `ans + 1`.
 * 5. Check the parity of the number of distinct characters:
 *    - If `ans + 1` is even, `ans` must be odd. The program checks `(ans & 1)` which is true for odd `ans`. Output: "CHAT WITH HER!".
 *    - If `ans + 1` is odd, `ans` must be even. The program checks `(ans & 1)` which is false for even `ans`. Output: "IGNORE HIM!".
 *
 * Time Complexity: O(L log L), where L is the length of the string, dominated by the sorting operation.
 * Space Complexity: O(L) to store the string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  // Sort the string to group identical characters together.
  sort(s.begin(), s.end());
  int l = s.length();
  int ans = 0; // This will count the number of changes between adjacent characters.
  /**
   * @brief Counts the number of times adjacent characters are different.
   * In a sorted string, this is equivalent to counting the number of distinct
   * characters minus one.
   * Pre-condition: The string `s` is sorted.
   * Invariant: `ans` stores the number of character changes found in the prefix of length `i`.
   */
  for (int i = 1; i < l; i++) {
    if (s[i] != s[i - 1])
      ans++;
  }
  //    cout<<s<<endl;
  //    cout<<ans<<endl;
  
  // The actual number of distinct characters is `ans + 1`.
  // The problem is to check the parity of `ans + 1`.
  // If `ans + 1` is even (female), `ans` is odd.
  // If `ans + 1` is odd (male), `ans` is even.
  /**
   * @brief Check the parity of `ans` to determine the gender.
   * `ans & 1` is a bitwise check for oddness.
   * If `ans` is odd, it means there's an even number of distinct characters.
   */
  if (ans & 1)
    cout << "CHAT WITH HER!" << endl; // Even number of distinct chars -> female
  else
    cout << "IGNORE HIM!" << endl; // Odd number of distinct chars -> male

  return 0;
}
