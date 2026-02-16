/**
 * @file 116A.CPP
 * @brief C-style solution for Codeforces problem 116A, "Tram".
 *
 * @description
 * This program finds the minimum required capacity of a tram. The tram makes `n`
 * stops, and at each stop, a certain number of passengers exit and enter.
 * The program simulates this process to find the peak number of passengers
 * on the tram at any point, which corresponds to the required capacity.
 *
 * @algorithm
 * 1. Initialize `k` (maximum capacity needed) and `S` (current occupancy) to 0.
 * 2. Read the total number of stops, `n`.
 * 3. Loop `n` times, once for each stop:
 *    a. Read the number of passengers exiting (`a`) and entering (`b`).
 *    b. Update the current occupancy: `S = S - a + b`.
 *    c. If the current occupancy `S` is greater than the max `k` seen so far,
 *       update `k` to this new maximum.
 * 4. After the loop, print the final value of `k`.
 *
 * @time_complexity O(N), where N is the number of stops.
 * @space_complexity O(1).
 */
#include <stdio.h>
int main() {
  // n: number of stops.
  // i: loop counter.
  // k: tracks the maximum capacity required.
  // S: tracks the current number of passengers (seats occupied).
  // a: passengers exiting.
  // b: passengers entering.
  int n, i, k = 0, S = 0, a, b;
  scanf("%d", &n);

  // Block Logic: Simulate the passenger flow over 'n' stops.
  for (i = 0; i < n; i++) {
    scanf("%d%d", &a, &b);
    // Update the current occupancy.
    S = S - a;
    S = S + b;
    // If the current occupancy is a new maximum, update the capacity counter.
    if (k <= S)
      k = S;
  }
  printf("%d", k);
  return 0;
}
