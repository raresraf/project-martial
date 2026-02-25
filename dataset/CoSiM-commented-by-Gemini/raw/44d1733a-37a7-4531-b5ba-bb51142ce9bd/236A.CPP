/**
 * @file raw/44d1733a-37a7-4531-b5ba-bb51142ce9bd/236A.CPP
 * @brief Solution to Codeforces problem 236A - Boy or Girl.
 *
 * @algorithm
 * The problem's decision logic is based on the number of distinct characters in a
 * given username. If the count of distinct characters is odd, the user is
 * identified as male ("IGNORE HIM!"). If it's even, the user is female
 * ("CHAT WITH HER!").
 *
 * This solution uses a `std::set<char>` to efficiently count the number of
 * unique characters in the input string. A set is a natural choice for this
 * task as it only stores unique elements.
 *
 * The algorithm is as follows:
 * 1. A `std::set<char>` named `m` is created.
 * 2. The code reads the input username character by character.
 * 3. Each character is inserted into the set `m`. The set automatically handles
 *    uniqueness, so duplicate characters are ignored.
 * 4. After all characters are processed, the size of the set (`m.size()`) gives
 *    the total count of distinct characters.
 * 5. The parity of this count is checked. If it's even, "CHAT WITH HER!" is
 *    printed. Otherwise, "IGNORE HIM!" is printed.
 *
 * Note: The input reading loop `while(cin >> x)` along with the newline check
 * is unconventional for reading a single string, as `cin >> x` skips whitespace.
 * However, it works for a single-word input.
 *
 * @complexity
 * - Time Complexity: O(L * log(K)), where L is the length of the string and K
 *   is the number of unique characters (the size of the set). For each
 *   character, insertion into a set takes logarithmic time. Since K is at most
 *   26 for lowercase English letters, this is effectively O(L).
 * - Space Complexity: O(K), where K is the number of unique characters, to
 *   store the elements in the set. In the worst case (all characters are
 *   unique), this is O(L), but for this problem, it's at most O(26).
 */
#include <bits/stdc++.h>

// Macro to disable synchronization with C stdio and untie cin/cout,
// for faster I/O in competitive programming.
#define _                                                                      \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);

using namespace std;

int main() {
  // Apply the fast I/O macro.
  _ 
  
  // s is declared but not used.
  string s;
  // `m` is a set to store the distinct characters of the username.
  set<char> m;
  char x;

  /**
   * @brief Read characters from input and insert them into the set.
   * The `cin >> x` idiom skips whitespace. The `if (x == '\n')` check
   * will not work as intended but the loop will terminate on EOF.
   * For typical competitive programming inputs (a single string), this works.
   */
  while (cin >> x) {
    if (x == '\n')
      break;
    m.insert(x);
  }

  /**
   * @brief Check the parity of the number of distinct characters.
   */
  if (m.size() % 2 == 0) {
    cout << "CHAT WITH HER!\n";
  } else {
    cout << "IGNORE HIM!\n";
  }

  return 0;
}
