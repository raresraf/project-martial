/**
 * @file 116A.CPP
 * @brief Solution to the "Tram" problem (116A) from Codeforces.
 * @details The program calculates the minimum possible capacity of a tram. It is given the
 * number of passengers exiting and entering at each stop, and it must find the
 * maximum number of passengers on the tram at any single point in time.
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm This solution simulates the passenger flow stop by stop.
 * 1. Read the total number of stops (n).
 * 2. Initialize a variable `curr_passengers` to 0 to track the current passenger count.
 * 3. Initialize a variable `max_passenger` to track the peak passenger count.
 * 4. Loop 'n' times, once for each stop.
 * 5. In each iteration, read the number of exiting (`a`) and entering (`b`) passengers.
 * 6. Update the current passenger count: `curr_passengers = curr_passengers - a + b`.
 * 7. After the update, compare `curr_passengers` with `max_passenger` and update
 *    `max_passenger` if the current count is higher.
 * 8. After the loop, `max_passenger` will hold the minimum required capacity, which is then printed.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of stops. The program iterates through each stop exactly once.
 * Space Complexity: O(1), as a constant amount of memory is used regardless of the number of stops.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {

  int a, b;
  int n;

  int max_passenger = -1;
  int curr_passengers = 0;
  ;

  scanf("%d", &n);

  /**
   * @block
   * @brief Simulates passenger flow over 'n' stops.
   * @invariant `curr_passengers` holds the number of people on the tram after the current stop's exchange.
   *            `max_passenger` holds the maximum value `curr_passengers` has reached so far.
   * @pre 'n' is the number of stops to simulate.
   * @post `max_passenger` contains the minimum required tram capacity.
   */
  while (n--) {

    scanf("%d%d", &a, &b);

    // Update current passenger count based on who gets off and who gets on.
    curr_passengers -= a;
    curr_passengers += b;

    // If the current number of passengers exceeds the max seen so far, update the max.
    if (max_passenger < curr_passengers)
      max_passenger = curr_passengers;
  }

  // The maximum number of passengers seen at any one time is the minimum required capacity.
  cout << max_passenger << endl;
}