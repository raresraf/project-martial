/**
 * @file 116A.CPP
 * @brief Calculates the minimum required capacity of a tram.
 *
 * This program determines the minimum capacity a tram needs to have to accommodate
 * passengers over a series of stops. It reads the number of passengers exiting
 * and entering at each stop and finds the maximum number of people inside the
 * tram at any given time.
 *
 * @b Algorithm: The program iteratively calculates the net change in passengers
 *    at each stop. It maintains a running total of the number of passengers (`c`)
 *    on the tram and keeps track of the highest number encountered (`m`).
 *
 * @b Time_Complexity: O(N), where N is the number of tram stops, because it
 *    iterates through the stops once.
 * @b Space_Complexity: O(1), as it only uses a few variables to store the counts
 *    and inputs for each stop.
 */
#include <cstdio>

int main() {
  int n, a, b, c(0), m(0); // n:stops, a:exits, b:enters, c:current capacity, m:max capacity
  scanf("%d", &n);
  
  /**
   * @brief This loop processes each stop.
   * It updates the current number of passengers and checks if it exceeds the
   * current maximum.
   */
  while (n--) {
    scanf("%d%d", &a, &b);
    c -= a; // Passengers exit.
    c += b; // Passengers enter.
    if (c > m) {
      m = c; // Update maximum capacity if needed.
    }
  }

  printf("%d\n", m);
  return 0;
}
