/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem on Codeforces.
 *
 * @details This program calculates the minimum required capacity of a tram by
 * simulating the change in passenger numbers at each stop and tracking the
 * peak number of passengers.
 *
 * Algorithm:
 * 1. Read the total number of stops, n.
 * 2. Initialize `cur` (current passengers) and `ma` (maximum passengers) to 0.
 * 3. Loop 'n' times, for each stop:
 *    a. Read the number of passengers exiting ('a') and entering ('b').
 *    b. Update the current passenger count: `cur = cur - a + b`.
 *    c. Update the maximum capacity `ma` if `cur` is greater than the current `ma`.
 * 4. Print the final maximum capacity `ma`.
 *
 * Time Complexity: O(N), where N is the number of stops.
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;
int main() {
  int n, a, b, ma = 0, cur = 0;
  scanf("%d", &n);
  /**
   * @brief Loop through each stop to find the maximum passenger count.
   * Pre-condition: `n` holds the number of stops. `cur` and `ma` are 0.
   * Invariant: `cur` holds the passenger count after the current stop's changes,
   * and `ma` holds the peak count seen up to this point.
   */
  // The 'register' keyword is a hint to the compiler and is often ignored by modern compilers.
  for (register int i = 0; i < n; i++) {
    scanf("%d", &a);
    scanf("%d", &b);
    // Update the current number of passengers.
    cur = cur - a;
    cur = cur + b;
    // Track the maximum number of passengers seen so far.
    ma = max(cur, ma);
  }
  printf("%d", ma);
}