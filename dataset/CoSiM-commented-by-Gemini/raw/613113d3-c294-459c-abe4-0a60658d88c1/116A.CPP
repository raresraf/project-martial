/**
 * @file 116A.CPP
 * @brief Solution for the "Tram" problem.
 *
 * @problem A tram makes 'n' stops. At each stop, a number of passengers exit
 * and another number enter. Given these numbers for each stop, the task is to
 * find the minimum capacity the tram must have.
 *
 * @algorithm The problem is equivalent to finding the maximum number of passengers
 * inside the tram at any given time. The algorithm simulates the process stop-by-stop:
 * 1. It reads all the passenger exit (a[i]) and entry (b[i]) numbers.
 * 2. It calculates the number of passengers in the tram after each stop.
 * 3. These passenger counts are stored in an array `ans`.
 * 4. Finally, it sorts the `ans` array to find the maximum value, which is the
 *    minimum required capacity.
 *
 * @time_complexity O(N log N) due to the sorting step. A more optimal O(N)
 * solution could find the maximum in a single pass without sorting.
 * @space_complexity O(N) to store the inputs and the passenger counts at each stop.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, i, a[1000], b[1000], x, t, ans[1000];
  cin >> n;

  // Read all inputs for the n stops first.
  for (i = 0; i < n; i++) {
    cin >> a[i] >> b[i];
  }

  // Initialize the passenger count after the first stop.
  // The problem implies a[0] is 0, as the tram starts empty.
  x = (0 - a[0]) + b[0]; // Current passengers after stop 0
  t = x;                 // `t` tracks the current total passengers.
  ans[0] = t;            // Store the passenger count for this stop.

  // Simulate the remaining stops.
  for (i = 1; i < n; i++) {
    // Update passenger count: (current - exiting) + entering.
    x = (t - a[i]) + b[i];
    ans[i] = x;
    t = x; // Update the current total for the next iteration.
  }

  // Sort the array of passenger counts to find the maximum.
  sort(ans, ans + n);
  // The last element in the sorted array is the maximum value.
  cout << ans[n - 1];

  return 0;
}
