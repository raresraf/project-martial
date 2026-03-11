/**
 * @file 160A.CPP
 * @brief Solution to the "Twins" problem (Codeforces 160A).
 * @details The problem asks for the minimum number of coins to take from a set
 *          such that the sum of the values of the taken coins is strictly
 *          greater than the sum of the values of the remaining coins.
 *
 * @algorithm This is a classic greedy algorithm problem. To minimize the number of
 *            coins taken while maximizing their sum, we should always pick the
 *            coins with the highest value first.
 *            1. Calculate the total sum of all coin values.
 *            2. Sort the coins in non-decreasing order.
 *            3. Iterate from the largest coin downwards, accumulating their sum and
 *               counting the number of coins taken.
 *            4. Stop when the accumulated sum is strictly greater than the sum of
 *               the remaining coins.
 *
 * @time_complexity O(N log N) due to the sorting of the coin values.
 * @space_complexity O(N) to store the coin values.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int step = 0;   // Counts the number of coins taken.
  int sumt = 0;   // Initially stores the total sum, then the sum of remaining coins.
  int n;
  int sumMe = 0;  // Stores the sum of coins taken by "me".
  int i;
  scanf("%d", &n);
  int A[n];
  /**
   * @brief Reads n integers into array A and calculates their total sum.
   * This is a compact for-loop often used in competitive programming.
   * The third expression of the for loop `sumt += A[i++]` is executed after the loop body.
   */
  for (i = 0; i < n; scanf(" %d", (A + i)), sumt += A[i++])
    ;
  // for (i = 0; i < n; printf("%d ", A[i++]));
  // printf("%d %d\n", sumt, sumMe);
  // Sort the coins in ascending order to apply the greedy strategy.
  sort(A, A + n);
  // for (i = 0; i < n; printf("%d ", A[i++]));
  // printf("\n");
  i = n - 1;
  /**
   * @brief Greedily take the largest coins until our sum is greater.
   * @invariant `sumMe` is the sum of `step` largest coins.
   * @invariant `sumt` is the sum of the remaining `n-step` coins.
   * The loop continues as long as the sum of coins taken is not strictly
   * greater than the sum of the remaining coins.
   */
  while (i >= 0 && sumMe <= sumt) {
    sumt -= A[i];
    sumMe += A[i];
    ++step;
    i--;
  }
  printf("%d\n", step);
  return 0;
}
