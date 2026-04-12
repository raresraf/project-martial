/**
 * @file 1343C.CPP
 * @brief Greedy algorithm for maximizing the sum of a maximal alternating subsequence.
 * 
 * Logic: To achieve the maximum length, one element must be selected from every 
 * contiguous block of elements having the same sign. To maximize the sum, 
 * the largest element within each such block is chosen.
 * 
 * Algorithm: Single-pass greedy block-wise maximization.
 * Time Complexity: $O(N)$ where N is the sequence length.
 * Space Complexity: $O(1)$ auxiliary space beyond input.
 */

#include <bits/stdc++.h>
#define ll long long
using namespace std;

/**
 * @brief Functional Utility: Sign difference detector.
 * @return True if 'a' and 'b' have opposing signs, False otherwise.
 * Logic: Compares the absolute difference of magnitudes vs the difference 
 * of values to determine sign alignment.
 */
bool dif(ll a, ll b) { return abs(abs(a) - abs(b)) != abs(a - b); }

int main() {
  int t;
  // Reads total independent test cases.
  scanf(" %d", &t);
  
  while (t--) {
    int n;
    scanf(" %d", &n);
    
    // Tracks the maximum element within the current contiguous sign-block.
    ll current_block_max = 0;
    // Accumulator for the final maximal alternating sum.
    ll total_sum = 0;

    /**
     * Block Logic: Iteratively processes the sequence to identify sign transitions.
     * Invariant: current_block_max stores the peak value of the current monotonic sign segment.
     */
    for (int i = 0; i < n; ++i) {
      ll va;
      scanf(" %lld", &va);
      
      /**
       * Case 1: Sign transition detected or initialization phase.
       * Logic: Commits the peak of the previous block to total_sum and starts a new block.
       */
      if (dif(current_block_max, va) || i == 0) {
        total_sum += current_block_max; 
        current_block_max = va;
      } else {
        /**
         * Case 2: Continuation of the current sign block.
         * Logic: Greedily updates the block peak.
         */
        current_block_max = max(current_block_max, va);
      }
    }
    
    // Output: Sum of all selected block peaks.
    printf("%lld\n", total_sum + current_block_max);
  }
  return 0;
}
