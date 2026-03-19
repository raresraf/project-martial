/*
 * 	User: Isanchez_Aguilar
 * 	Problem: CodeForces 266A - Stones on the Table
 */
/**
 * @brief Solves the "Stones on the Table" problem.
 * @details The program calculates the minimum number of stones to remove from a row
 * such that no two adjacent stones have the same color.
 *
 * Algorithm: Greedy Single-Pass
 * The code iterates through the stones and counts any stone that is the same color
 * as the one immediately preceding it. This is a greedy approach.
 * The implementation uses a nested loop structure.
 *
 * NOTE: The inner loop `while (s[i - 1] == s[i])` does not check if `i` is within the
 * bounds of the string, which can lead to an out-of-bounds read if the last two or more
 * stones are identical. However, it may pass certain test environments if string access
 * beyond its length is handled gracefully (e.g., returns a null character).
 *
 * Time Complexity: O(N), where N is the number of stones, as each stone is visited once.
 * Space Complexity: O(N) to store the input string of stones.
 */
#include <bits/stdc++.h>

using namespace std;

int main(void) {
  // Functional Utility: Optimizes C++ stream I/O for faster execution in competitive programming.
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);

  // n: the number of stones on the table.
  int n;
  // s: a string representing the colors of the stones.
  string s;
  cin >> n >> s;

  // ans: the count of stones to be removed.
  int ans = 0;

  // i: the loop counter, starting from the second stone.
  int i = 1;

  /**
   * Block Logic: Iterate through the string to find adjacent stones of the same color.
   * Pre-condition: 'i' points to the current stone being compared with the previous one.
   * The loop continues as long as 'i' is within the bounds of the string.
   */
  while (i < n) {
    /**
     * Block Logic: Count consecutive identical stones.
     * Pre-condition: s[i-1] and s[i] are adjacent stones.
     * Invariant: This loop increments 'ans' for each additional stone in a sequence
     * of identical colors, and advances 'i' to the end of that sequence.
     * WARNING: This loop does not check `i < n`, creating a potential out-of-bounds read
     * on `s[i]` if a sequence of identical stones extends to the end of the string.
     */
    while (s[i - 1] == s[i]) {
      // If the current stone has the same color as the previous one, increment the removal count.
      ++ans;
      // Move to the next stone to continue checking the sequence.
      ++i;
    }
    
    // Move to the next position to start a new comparison.
    ++i;
  }

  // Output the total number of stones to remove.
  cout << ans << "\n";

  return 0;
}
