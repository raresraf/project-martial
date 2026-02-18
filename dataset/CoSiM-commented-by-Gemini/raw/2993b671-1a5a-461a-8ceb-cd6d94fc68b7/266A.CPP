/**
 * @file 266A.CPP
 * @brief Solution to the "Stones on the Table" problem (266A) from Codeforces.
 * @details The problem asks for the minimum number of stones to remove from a row
 * of colored stones such that no two adjacent stones have the same color. This is
 * equivalent to counting the number of adjacent pairs with the same color.
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm This solution attempts to solve the problem by iterating through the
 * string and counting pairs of identical adjacent characters.
 * 1. Read the number of stones (n) and the string of colors (s).
 * 2. Initialize a counter `ans` to 0.
 * 3. Iterate through the string from the first character to the last.
 * 4. At each position `i`, check if the character `s[i]` is the same as the next one, `s[i+1]`.
 * 5. If they are the same, increment `ans`.
 * 6. Print the final count.
 *
 * @note There is a bug in the loop condition. The loop runs up to `i = n-1`, which
 * causes an out-of-bounds memory access when `s[i+1]` (i.e., `s[n]`) is checked.
 * The loop should only run up to `n-2`.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of stones, due to the single pass.
 * Space Complexity: O(N), to store the input string.
 */
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  string s;
  cin >> s;
  int ans = 0;
  /**
   * @block
   * @brief Iterates through the string to count adjacent identical characters.
   * @warning The loop condition `i <= n - 1` leads to an out-of-bounds access on
   *          the last iteration when it checks `s[i+1]`. The loop should be `i < n - 1`.
   */
  for (int i = 0; i <= n - 1; i++) {
    if (s[i] == s[i + 1])
      ans += 1;
  }
  cout << ans << '\n';
}