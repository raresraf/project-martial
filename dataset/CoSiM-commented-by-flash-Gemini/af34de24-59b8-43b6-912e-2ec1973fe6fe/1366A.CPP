/**
 * @file 1366A.CPP
 * @brief Logic for maximizing item production under dual resource constraints.
 * 
 * Logic: Calculates the maximum number of items (shovels and swords) that can be crafted 
 * from 'a' sticks and 'b' diamonds. Each item requires a total of 3 units (either 
 * 2:1 or 1:2 ratio of sticks to diamonds).
 * 
 * Algorithm: Closed-form optimization based on resource bottleneck analysis.
 * Formula: Max items = min(a, b, (a + b) / 3).
 * 
 * Time Complexity: $O(T)$ where T is the number of test cases.
 * Space Complexity: $O(1)$ constant space.
 */

#include <bits/stdc++.h>
#define ll long long
using namespace std;

int main() {
  ll a, b, t;
  // Reads total independent crafting scenarios.
  cin >> t;

  /**
   * Block Logic: Evaluates each scenario for maximum crafting capacity.
   * Invariant: Total items are bounded by the total resource sum / 3 and 
   * the individual category constraints.
   */
  while (t--) {
    cin >> a >> b;
    
    /**
     * Functional Utility: Resource-based capacity analysis.
     */
    if (a == 0 || b == 0)
      // Logic: No items can be made if either resource is missing.
      cout << "0" << endl;
    else if (2 * a < b)
      /**
       * Case 1: Sticks (a) are the severe bottleneck.
       * Even using the 1:2 ratio for all items, we are limited by 'a'.
       */
      cout << a << endl;
    else if (2 * b < a)
      /**
       * Case 2: Diamonds (b) are the severe bottleneck.
       */
      cout << b << endl;
    else
      /**
       * Case 3: Balanced resources.
       * Max items is limited by the total headcount of available units divided by 3.
       */
      cout << (a + b) / 3 << endl;
  }
  
  return 0;
}
