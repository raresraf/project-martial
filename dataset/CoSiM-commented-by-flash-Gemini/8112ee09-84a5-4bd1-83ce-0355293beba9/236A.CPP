/**
 * @file 236A.CPP
 * @brief Solution to Codeforces problem 236A - Boy or Girl.
 *        This program determines if a given username belongs to a "boy" or a "girl"
 *        based on the parity (even or odd) of the number of distinct characters in the username.
 *        If the number of distinct characters is even, it's a "girl" ("CHAT WITH HER!").
 *        If the number of distinct characters is odd, it's a "boy" ("IGNORE HIM!").
 *        This implementation efficiently counts distinct characters using a `std::set`.
 *
 * Algorithm: Counting Distinct Characters using a Set
 *   1. Reads the input username string `s`.
 *   2. Initializes an empty `std::set<char>` to store unique characters.
 *   3. Iterates through each character of the input string `s`:
 *      a. Inserts the current character into the `std::set`. The set automatically
 *         handles uniqueness, so duplicate insertions have no effect on size.
 *   4. After iterating through the entire string, the size of the `std::set`
 *      (`st.size()`) gives the total number of distinct characters.
 *   5. Checks the parity of this count:
 *      a. If `st.size()` is even, prints "CHAT WITH HER!".
 *      b. If `st.size()` is odd, prints "IGNORE HIM!".
 *
 * Time Complexity: O(L log C), where L is the length of the input string and C
 *                  is the size of the character set (at most 26 for English alphabet).
 *                  Insertion into a `std::set` takes logarithmic time with respect to its size.
 *                  Effectively O(L) as C is a small constant.
 * Space Complexity: O(C) for storing the distinct characters in the `std::set`.
 *                   Effectively O(1) as C is a constant. O(L) for storing the input string `s`.
 */

#include <bits/stdc++.h> // Includes common C++ standard library headers like iostream, string, set.
using namespace std;

int main() {
  /**
   * @brief Main function that solves the "Boy or Girl" problem.
   *
   * Reads a username, counts its distinct characters using a `std::set`,
   * and prints the appropriate message based on the parity of the count.
   *
   * @return int Returns 0 upon successful execution.
   */
  string s; // Input username string.
  cin >> s;
  set<char> st; // `std::set` to store unique characters.
  int i;         // Loop counter.
  int len = s.length(); // Length of the input string.
  /**
   * Block Logic: Iterates through the input string and inserts each character into the set.
   * The set automatically handles uniqueness, ensuring only distinct characters are stored.
   * Invariant: `st` contains all unique characters from `s` processed so far.
   */
  for (i = 0; i < len; i++) {
    st.insert(s[i]); // Inserts the character; duplicates are ignored.
  }
  int a = st.size(); // `a` now holds the count of distinct characters.
  /**
   * Conditional Logic: Checks the parity of the count of distinct characters.
   * Prints "CHAT WITH HER!" if even, "IGNORE HIM!" if odd.
   */
  if (a % 2 == 0) // If the number of distinct characters is even.
    printf("CHAT WITH HER!\n");
  else // If the number of distinct characters is odd.
    printf("IGNORE HIM!\n");
  return 0;
}
