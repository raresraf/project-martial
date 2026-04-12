/**
 * @file 160A.CPP
 * @brief Greedy algorithm for partitioning a set to achieve a strictly greater sum.
 * 
 * Logic: Identifies the minimum number of elements (coins) whose sum exceeds 
 * half of the total set value. Employs a greedy selection strategy by prioritizing 
 * elements with the highest values.
 * 
 * Algorithm: Sorting followed by a suffix-sum based greedy selection.
 * Time Complexity: $O(N \log N)$ due to sorting.
 * Space Complexity: $O(N)$ to store the set of values.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, total_sum = 0, current_partition_sum = 0, count = 0;
  cin >> n;
  int arr[n];

  /**
   * Block Logic: Populates the value array and calculates the global aggregate.
   * Invariant: total_sum represents the sum of all n elements in the set.
   */
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    total_sum += arr[i];
  }

  // Logic: Reorders the set to enable greedy selection from the highest value downwards.
  sort(arr, arr + n);

  /**
   * Block Logic: Greedily accumulates elements from the top of the sorted range.
   * Invariant: Selection continues until the current_partition_sum strictly exceeds 
   * the threshold (total_sum / 2).
   */
  for (int i = n - 1; i >= 0; i--) {
    current_partition_sum += arr[i];
    count++;
    
    /**
     * Functional Utility: Early termination once the partition criteria is met.
     * Logic: sum / 2 is the integer floor; the condition correctly identifies 
     * a sum strictly greater than half.
     */
    if (current_partition_sum > total_sum / 2) {
      cout << count;
      break;
    }
  }

  return 0;
}
