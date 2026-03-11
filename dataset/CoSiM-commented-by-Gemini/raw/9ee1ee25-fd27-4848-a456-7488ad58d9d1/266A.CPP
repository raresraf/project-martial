/**
 * @file 266A.CPP
 * @brief Solves the "Stones on the Table" problem.
 *
 * This program calculates the minimum number of stones to remove from a row
 * of colored stones such that no two adjacent stones have the same color.
 *
 * Algorithm: The code iterates through the sequence of stones, identifying and
 *            counting consecutive groups of the same color. For each group of
 *            size 'k' where k > 1, it determines that k-1 stones must be removed.
 *
 * Time Complexity: O(N), where N is the number of stones, as each stone is visited once.
 * Space Complexity: O(N) to store the string of stone colors.
 */
#include <bits/stdc++.h>
using namespace std;
#define f first
#define s second
#define mp make_pair
#define pb push_back
typedef long long ll;

int main() {
  // freopen("input.txt", "r", stdin);
  ios_base::sync_with_stdio(false); // Optimizes C++ stream I/O.
  
  int n; // The total number of stones.
  int ans = 0; // The total number of stones to be removed.
  cin >> n;
  
  string xs; // Represents the sequence of stone colors.
  cin >> xs;
  
  char init; // Stores the color of the current group of stones.

  /**
   * @brief Main loop to iterate through the stones and count removals.
   *
   * Pre-condition: 'xs' contains the string of stone colors.
   * Invariant: At the start of each iteration, 'i' is the index of the first stone
   *            in a new potential group of same-colored stones.
   */
  for (int i = 0; i < xs.size(); i++) {
    int count = 0; // Counter for the number of stones in the current consecutive group.
    init = xs[i];
    bool inwhile = false; // Flag to check if the inner while loop was entered.

    /**
     * @brief Inner loop to count consecutive stones of the same color.
     * Logic: This loop advances the main iterator 'i' as long as the stones
     *        have the same color, counting them.
     */
    while (i < xs.size() && xs[i] == init) {
      ++count;
      i++;
      inwhile = true;
    }
    
    // If a group of same-colored stones is found (size > 1), add the number
    // of stones to remove (count - 1) to the total answer.
    if (count > 1)
      ans += (count - 1);
      
    // The inner while loop increments 'i' one time too many (to check the next
    // character that breaks the sequence). This post-decrement compensates for it,
    // ensuring the outer for-loop's i++ correctly moves to the start of the next group.
    if (inwhile)
      --i;
  }
  
  // Output the minimum number of stones to remove.
  cout << ans << endl;
}
