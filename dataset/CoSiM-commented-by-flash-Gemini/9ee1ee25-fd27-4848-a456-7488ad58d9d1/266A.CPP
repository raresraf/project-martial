/**
 * @file 266A.CPP
 * @brief Discrete state-reduction algorithm to resolve adjacency constraints in a sequence.
 * 
 * Logic: Calculates the minimum number of deletions required so that no two adjacent 
 * elements have identical states (colors). This is a single-pass greedy reduction.
 * 
 * Time Complexity: $O(N)$ where N is the length of the input sequence.
 * Space Complexity: $O(N)$ for the sequence buffer.
 */

#include <bits/stdc++.h>
using namespace std;
#define f first
#define s second
#define mp make_pair
#define pb push_back
typedef long long ll;

int main() {
  /**
   * Functional Utility: Optimizes standard I/O throughput by decoupling C++ streams 
   * from C standard streams.
   */
  ios_base::sync_with_stdio(false);
  int n, ans = 0;
  cin >> n;
  string xs;
  cin >> xs;

  /**
   * Block Logic: Identifies contiguous segments of identical states.
   * Invariant: For each segment of length L, (L-1) elements must be removed 
   * to satisfy the non-adjacency constraint.
   */
  for (int i = 0; i < xs.size(); i++) {
    int count = 0;
    // Functional Utility: Captures the state of the current sequence element.
    char init = xs[i];
    bool inwhile = false;

    // Logic: Inner traversal to measure the extent of a single-color cluster.
    while (xs[i] == init) {
      ++count;
      i++;
      inwhile = true;
    }

    /**
     * Functional Utility: Accumulates the necessary deletions for the current cluster.
     * Logic: If a cluster has more than 1 stone, all but one must be removed.
     */
    if (count > 1)
      ans += (count - 1);
    
    // Inline: Adjusts the outer loop index to account for the lookahead in the while loop.
    if (inwhile)
      --i;
  }
  
  // Output the total number of deletions performed.
  cout << ans << endl;
}
