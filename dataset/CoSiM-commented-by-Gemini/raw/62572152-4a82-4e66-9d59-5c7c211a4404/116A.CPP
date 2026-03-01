/**
 * @file 116A.CPP
 * @brief Optimal solution for the "Tram" problem.
 *
 * @problem A tram makes 'n' stops. At each stop, a number of passengers exit
 * and another number enter. The tram starts empty. The task is to find the
 * minimum capacity the tram must have to never be overcrowded.
 *
 * @algorithm This code uses an optimal O(N) time and O(1) space approach. It
 * simulates the process by iterating through the stops and processing the data
 * on the fly.
 * 1. Two integer variables are maintained: `res` (current passengers) and `max`
 *    (maximum passengers seen so far).
 * 2. The code loops 'n' times, once for each stop.
 * 3. In each iteration, it reads the number of exiting (`a`) and entering (`b`)
 *    passengers.
 * 4. It updates the `res` variable by subtracting `a` and adding `b`.
 * 5. It then updates `max` if the new `res` is greater than the current `max`.
 * 6. After the loop, `max` holds the peak passenger count, which is the answer.
 *
 * @time_complexity O(N), where N is the number of stops.
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  int a, b;
  // `res` tracks the current number of passengers on the tram.
  int res = 0;
  // `max` tracks the maximum number of passengers seen at any point.
  int max = -1;

  // Loop through each of the 'n' stops.
  while (n--) {
    cin >> a >> b;
    // The check for `a == 0` is redundant, as `res - 0 + b` is the same as `res + b`.
    // It literally models the first stop having 0 exits.
    if (a == 0)
      res = res + b;
    else
      // For all other stops, update the current passenger count.
      res = res - a + b;
    
    // If the current passenger count is the highest seen so far, update max.
    if (res > max)
      max = res;
  }
  // The answer is the maximum passenger count recorded.
  cout << max;
}
