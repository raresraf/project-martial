/**
 * @file 236A.CPP
 * @brief Solution to Codeforces problem 236A - Boy or Girl.
 *
 * @algorithm
 * The problem's decision logic is based on the number of distinct characters in a
 * given username. If the count of distinct characters is odd, the user is
 * identified as male ("IGNORE HIM!"). If the count is even, the user is female
 * ("CHAT WITH HER!").
 *
 * This implementation uses a `std::set` to efficiently count the distinct characters:
 * 1. Read the username string `s`.
 * 2. Create an empty `std::set<char>`.
 * 3. Use a range-based for loop to iterate through the input string and insert
 *    each character into the set. `std::set` automatically handles uniqueness.
 * 4. After the loop, the size of the set gives the number of distinct characters.
 * 5. Check if this size is odd or even using the modulo operator (`% 2`).
 * 6. Print the appropriate message based on the parity.
 *
 * @complexity
 * - Time Complexity: O(L * log(K)), where L is the length of the string and K is
 *   the number of distinct characters (a small constant). This is effectively O(L).
 * - Space Complexity: O(K) to store the distinct characters in the set, which is
 *   effectively O(1) as K is bounded by the character set size.
 */
// author @Nishant

#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  cin >> s;
  set<char> un;
  /**
   * @brief Populates a set with the characters from the username.
   * The range-based for loop provides a clean syntax for iterating through the string.
   * The size of the set will then equal the number of distinct characters.
   */
  for (char x : s) {
    un.insert(x);
  }
  /**
   * @brief Determines the output based on the parity of the distinct character count.
   * If the count is odd, output "IGNORE HIM!".
   * If the count is even, output "CHAT WITH HER!".
   */
  if (un.size() % 2) {
    cout << "IGNORE HIM!";
  } else {
    cout << "CHAT WITH HER!";
  }
  return 0;
}