/**
 * @file 160A.CPP
 * @brief Greedy algorithm for set partitioning to achieve a strictly greater sum.
 * 
 * Logic: Calculates the minimum number of elements (coins) whose sum is strictly 
 * greater than the sum of the remaining elements. Employs a greedy strategy 
 * by selecting elements with the highest values first.
 * 
 * Algorithm: Sorting followed by a suffix-sum based greedy accumulation.
 * Time Complexity: $O(N \log N)$ due to sorting.
 * Space Complexity: $O(N)$ for the global element buffer.
 */

#include <bits/stdc++.h>
#include <iostream>
#include <math.h>

using namespace std;

// Data Structure: Global vector to store element values across multiple scenarios.
vector<int> arr;

int main() {
  int n;
  int total_aggregate;
  int input_val;
  int partition_sum;

  /**
   * Block Logic: Processes independent test scenarios from standard input.
   */
  while (cin >> n) {
    total_aggregate = 0;
    arr.clear(); // Ensure fresh state for each scenario.

    /**
     * Block Logic: Population and total sum calculation.
     * Invariant: total_aggregate represents the sum of all elements in the set.
     */
    for (int i = 0; i < n; i++) {
      cin >> input_val;
      arr.push_back(input_val);
      total_aggregate += input_val;
    }

    // Logic: Reorders the set to facilitate greedy selection from largest to smallest.
    int n_elements = (int)arr.size();
    sort(arr.begin(), arr.end());

    partition_sum = 0;
    int index = n_elements;

    /**
     * Block Logic: Greedy accumulation loop.
     * Logic: Repeatedly picks the largest available element until the sum of 
     * picked elements exceeds the sum of unpicked elements.
     * Invariant: picked_sum > (total_aggregate - picked_sum).
     */
    do {
      partition_sum += arr[--index];
    } while (partition_sum <= (total_aggregate - partition_sum) && index > 0);

    /**
     * Functional Utility: Calculates the count of selected elements.
     * Result: Total elements minus the remaining index gives the selection cardinality.
     */
    cout << n_elements - index << endl;
  }
  
  return 0;
}
