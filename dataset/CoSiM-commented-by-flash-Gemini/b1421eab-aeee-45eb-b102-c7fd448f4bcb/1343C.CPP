/**
 * @file 1343C.CPP
 * @brief Greedy algorithm for maximizing the sum of an alternating sign subsequence.
 * 
 * Logic: To maximize subsequence length and sum, the algorithm partitions the 
 * input sequence into contiguous segments of identical sign parity. From each 
 * segment, it greedily selects the element with the maximum algebraic value.
 * 
 * Algorithm: Contiguous sign-block partitioning with local peak selection.
 * Time Complexity: $O(N)$ where N is the sequence length.
 * Space Complexity: $O(N)$ for the input vector.
 */

#include <bits/stdc++.h>
using namespace std;

typedef long long int ll;

int main() {
  /**
   * @brief Functional Utility: Sign parity detector.
   * @param x The numeric value to classify.
   * @return 1 for positive integers, -1 for negative integers.
   */
  auto sgn = [&](int x) {
    if (x > 0) {
      return 1;
    } else {
      return -1;
    }
  };

  int t;
  // Reads the total number of independent test scenarios.
  cin >> t;
  
  while (t--) {
    int n;
    // Accumulator for the resulting maximal alternating sum.
    ll total_sum = 0;
    cin >> n;
    
    // Data Structure: Buffer for the current sequence.
    std::vector<int> v(n);
    for (auto &it : v)
      cin >> it;

    /**
     * Block Logic: Iteratively traverses segments of identical sign.
     * Invariant: Each outer loop iteration processes one complete contiguous sign block.
     */
    for (int i = 0; i < n; ++i) {
      // Local Peak: Initialized with the first element of the current sign block.
      int current_block_max = v[i];
      int j = i;
      
      /**
       * Block Logic: Inner sweep to find the block's maximum value.
       * Condition: Continues as long as the sign parity remains unchanged.
       */
      while (j < n && sgn(v[i]) == sgn(v[j])) {
        // Logic: Greedily tracks the largest element in the current segment.
        current_block_max = max(current_block_max, v[j]);
        ++j;
      }
      
      // Functional Utility: Commits the optimal choice for this segment.
      total_sum += current_block_max;
      
      // Inline: Advances the outer pointer to the start of the next sign block.
      i = j - 1;
    }
    
    // Output: Sum of all selected segment peaks.
    cout << total_sum << "\n";
  }

  return 0;
}
