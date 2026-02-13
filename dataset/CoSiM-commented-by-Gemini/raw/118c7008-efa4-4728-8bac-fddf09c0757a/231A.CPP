/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem (Codeforces 231A).
 * @author competitive-programmer
 *
 * @details This program calculates the number of problems a three-person team
 * will implement. The condition for implementation is that at least two of the
 * three members are sure about the solution (represented by a '1').
 *
 * Algorithm (Sorting-based approach):
 * 1. Read the number of problems, `t`.
 * 2. Initialize a counter `k` to 0.
 * 3. Loop `t` times for each problem.
 * 4. Read the three opinions (0 or 1) into an array.
 * 5. Sort the array. After sorting an array of three 0s and 1s:
 *    - If there is zero or one '1', the middle element `a[1]` will be 0.
 *    - If there are two or three '1's, the middle element `a[1]` will be 1.
 * 6. Therefore, checking if the middle element `a[1]` is non-zero is a
 *    sufficient condition to know if at least two members are sure.
 * 7. If the condition is met, increment the counter `k`.
 * 8. Print the final count `k`.
 *
 * Time Complexity: O(t), as sorting a 3-element array is a constant time
 * operation, and we do this for each of the `t` problems.
 * Space Complexity: O(1).
 */
// link : https://codeforces.com/problemset/problem/231/A

#include <bits/stdc++.h>

using namespace std;

// t: number of problems, k: counter for problems to be implemented.
int t, k;

// Using `main()` without a return type is non-standard. `int main()` is preferred.
main() {

  cin >> t;

  /**
   * Block Logic: This loop iterates through each of the 't' problems.
   */
  while (t--) {
    // An array to hold the three opinions for the current problem.
    int a[3];

    cin >> a[0] >> a[1] >> a[2];

    // Sort the opinions. This will arrange the 0s and 1s in non-decreasing order.
    sort(a, a + 3);

    /**
     * Decision Logic: Check if at least two members are sure.
     * After sorting, if the middle element `a[1]` is 1, it guarantees that
     * `a[2]` is also 1, meaning at least two members are sure.
     * The check `a[2] != 0` is redundant but does not affect the correctness.
     */
    if (a[1] != 0 && a[2] != 0)
      k++;
  }

  cout << k;
}
