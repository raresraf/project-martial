/**
 * @file 266A.CPP
 * @brief An inefficient and overly complex solution to the "Stones on the Table" problem.
 * @details This program calculates the minimum number of stones to remove from a row
 * so that no two adjacent stones share the same color. It does so by iteratively
 * finding and removing a stone that is identical to one of its neighbors.
 *
 * @see https://codeforces.com/problemset/problem/266/A
 *
 * Algorithm:
 * The program iterates through the string, and if it finds a character that is
 * identical to either of its neighbors, it removes that character and counts it
 * as one removal. The loop index is then adjusted to re-evaluate the new character
 * at the same position.
 *
 * CRITICAL FLAW / INEFFICIENCY:
 * This approach is functionally correct for the problem, but it is highly
 * inefficient. Modifying the string with `s.erase()` inside the loop is an
 * O(N) operation. Since the loop can also run up to N times, the total time
 * complexity is O(N^2). The optimal solution simply counts adjacent identical
 * characters in a single pass (O(N) time) without modifying the string at all.
 *
 * Time Complexity: O(N^2)
 * Space Complexity: O(N)
 */
#include <bits/stdc++.h>
#define endl "\n"
#define lli long long int
#define ulli unsigned long long int

using namespace std;

int main() {
  // Fast I/O for competitive programming.
  ios_base::sync_with_stdio(false);
  cin.tie(0);

  int n;
  cin >> n;

  string s;
  cin >> s;

  int count = 0;
  /**
   * @brief Main loop to find and remove stones with identical neighbors.
   * This loop is inefficient due to modifying the string while iterating.
   */
  for (int i = 1; i < s.size() - 1; i++) {
    // This complex condition is equivalent to `if (s[i] == s[i-1] || s[i] == s[i+1])`.
    if (!(s[i] != s[i - 1] && s[i] != s[i + 1])) {
      count++;
      // Erasing an element from the middle of a string is an O(N) operation.
      s.erase(s.begin() + i);
      // Decrement 'i' to re-check the same index, which now holds a new character.
      i--;
    }
  }

  // An edge case check for a two-character string that might result from the loop.
  if (s.size() == 2 && s[0] == s[1])
    count++;

  cout << count << endl;
}
