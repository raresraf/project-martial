/**
 * @file 116A.CPP
 * @brief Calculates the minimum required capacity of a tram.
 *
 * This program simulates the passenger flow in a tram across several stops
 * to determine the minimum capacity required to accommodate all passengers
 * at any point in time.
 *
 * Algorithm:
 * The program iterates through each tram stop, updating the current number of
 * passengers by subtracting those who exit and adding those who enter. It
 * maintains a running maximum of the passenger count, which represents the
 * minimum required capacity.
 *
 * Time Complexity: O(n), where n is the number of tram stops.
 * Space Complexity: O(1), as it uses a constant amount of memory.
 */

#include <stdio.h>

using namespace std;

int main() {
  // n: the total number of tram stops.
  // a: the number of passengers exiting the tram at a stop.
  // b: the number of passengers entering the tram at a stop.
  // minTram: the minimum capacity required for the tram, initialized to 0.
  // totalPassengers: the current number of passengers on the tram.
  int n, a, b, minTram = 0, totalPassengers = 0;

  // Read the number of tram stops.
  scanf("%d", &n);

  /**
   * Block Logic: Iterate through each stop to simulate passenger flow.
   * Pre-condition: `n` contains the number of stops to process.
   * Invariant: `totalPassengers` holds the current passenger count, and
   * `minTram` holds the maximum passenger count seen so far.
   */
  while (n--) {
    // Read the number of passengers exiting and entering at the current stop.
    scanf("%d %d", &a, &b);

    // Update the total number of passengers.
    totalPassengers -= a;
    totalPassengers += b;

    // If the current number of passengers exceeds the previously recorded
    // maximum, update the maximum.
    if (totalPassengers > minTram)
      minTram = totalPassengers;
  }

  // Print the minimum required tram capacity.
  printf("%d\n", minTram);

  return 0;
}
