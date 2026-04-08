/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem from Codeforces (Problem 160A).
 * @details The goal is to find the minimum number of coins to take such that their
 * sum is strictly greater than the sum of the remaining coins. This is a
 * classic greedy algorithm problem.
 *
 * @algorithm
 * 1. Read the number of coins, `n`, and their values into an array `A`.
 * 2. Calculate the total sum of all coin values, `sumt`.
 * 3. Sort the array of coins `A` in non-decreasing order.
 * 4. Initialize the sum of coins taken (`sumMe`) and the count of coins taken (`step`) to 0.
 * 5. Iterate from the largest coin downwards. In each step:
 *    a. Move the current largest coin's value from the remaining sum (`sumt`) to the taken sum (`sumMe`).
 *    b. Increment the coin count (`step`).
 * 6. The loop continues as long as `sumMe` is not strictly greater than `sumt`.
 * 7. Print the final count `step`.
 *
 * @time_complexity O(N log N), dominated by the sorting step.
 * @space_complexity O(N) to store the coin values.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int step = 0; // Counts the number of coins taken.
  int sumt = 0; // Initially holds the total sum of all coins, then the sum of remaining coins.
  int n;
  int sumMe = 0; // Holds the sum of coins taken.
  int i;
  scanf("%d", &n);
  int A[n];

  // A compact loop to read coin values and calculate the total sum.
  for (i = 0; i < n; scanf(" %d", (A + i)), sumt += A[i++])
    ;
  
  // Sort the coins in ascending order to facilitate the greedy approach.
  sort(A, A + n);

  i = n - 1;
  /**
   * @block_logic Greedily take the largest coins until the sum of taken coins
   * is greater than the sum of the remaining ones.
   * @pre `A` is sorted, `sumt` is the total sum, `sumMe` is 0.
   * @invariant In each iteration, the `i`-th coin is moved from the "remaining" pile
   *            to the "taken" pile.
   * @post The loop terminates when `sumMe > sumt`. `step` holds the minimal number of coins.
   */
  while (i >= 0 && sumt >= sumMe) {
    sumt -= A[i]; // Decrease the remaining sum.
    sumMe += A[i]; // Increase the taken sum.
    ++step; // Increment the count of coins taken.
    i--;
  }

  printf("%d\n", step);
  return 0;
}
