/**
 * @file 96A.CPP
 * @brief Solution to the "Football" problem (96A) from Codeforces.
 * @details The program determines if a situation in a football match is "dangerous".
 * A situation is dangerous if there are at least 7 consecutive players from the same team.
 * Teams are represented by '0's and '1's.
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm The solution involves a single linear scan through the player string to
 * count consecutive identical characters.
 * 1. Read the string of players.
 * 2. Initialize a `count` for consecutive players to 1.
 * 3. Iterate through the string. At each position `i`, compare the player `s[i]` with the
 *    previous player `s[i-1]`.
 * 4. If they are the same, increment `count`. If `count` reaches 7, the situation is
 *    dangerous; print "YES" and terminate.
 * 5. If they are different, reset `count` to 1.
 * 6. If the loop completes without finding 7 consecutive players, the situation is not
 *    dangerous; print "NO".
 *
 * @note The loop starts at `i=0`, causing an out-of-bounds access `s[-1]` in the first
 * iteration. While this is undefined behavior, the logic works if `s[-1]` happens to not
 * equal `s[0]`. A robust implementation would start the loop at `i=1`.
 *
 * @complexity
 * Time Complexity: O(N), where N is the length of the string, due to the single pass.
 * Space Complexity: O(N), to store the input string.
 */
#include <iostream>
using namespace std;
int main() {
  string s;
  cin >> s;
  int count = 1;
  /**
   * @block
   * @brief Scans the string to find a sequence of 7 identical consecutive characters.
   * @invariant `count` holds the length of the current run of identical characters ending at `s[i-1]`.
   * @pre 's' contains the string of players.
   * @post Prints "YES" if a run of 7 is found, otherwise the loop completes.
   */
  for (int i = 0; i < s.length(); i++) {
    // NOTE: Accessing s[i-1] when i=0 is undefined behavior.
    // The intended logic compares the current character with the previous one.
    if (s[i] == s[i - 1]) {
      count += 1;
      if (count == 7) {
        cout << "YES" << endl;
        return 0;
      }
    } else
      count = 1;
  }
  cout << "NO" << endl;
  return 0;
}
