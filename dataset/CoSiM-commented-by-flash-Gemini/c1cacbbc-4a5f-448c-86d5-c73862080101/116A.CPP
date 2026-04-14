/**
 * @file 116A.CPP
 * @brief This program solves the "Tram" problem. It calculates the minimum capacity
 * a tram needs to have to accommodate all passengers during its journey.
 * This is equivalent to finding the maximum number of passengers simultaneously
 * present in the tram at any point.
 * Algorithm: Iteratively updates the current number of passengers and keeps track
 * of the maximum number of passengers encountered.
 * Time Complexity: O(N), where N is the number of stops, due to a single pass.
 * Space Complexity: O(1) for storing a few integer variables.
 */
#include <stdio.h> // Required for standard input/output functions like scanf and printf.

using namespace std; // Brings all names from the std namespace into the current scope.

int main() {
  int n; // Stores the number of tram stops.
  int a; // Stores the number of passengers exiting the tram at the current stop.
  int b; // Stores the number of passengers entering the tram at the current stop.
  int minTram = 0; // Stores the maximum number of passengers observed in the tram, representing the minimum required capacity.
  int totalPassengers = 0; // Stores the current number of passengers in the tram.

  // Reads the total number of tram stops 'n'.
  scanf("%d", &n);
  /**
   * Block Logic: Processes the events at each tram stop.
   * For each stop, it updates the `totalPassengers` by subtracting those exiting
   * and adding those entering. It then updates `minTram` if the current `totalPassengers`
   * exceeds the previously recorded maximum.
   * Invariant: `totalPassengers` reflects the current occupancy, and `minTram` holds the peak occupancy so far.
   */
  while (n--) { // Loop iterates 'n' times, once for each stop.
    // Reads the number of passengers exiting ('a') and entering ('b') at the current stop.
    scanf("%d %d", &a, &b);
    totalPassengers -= a; // Passengers exit.
    totalPassengers += b; // Passengers enter.
    /**
     * Conditional Logic: Updates `minTram` if the current number of passengers (`totalPassengers`)
     * is greater than the previously recorded maximum (`minTram`).
     * `minTram` will ultimately hold the maximum occupancy reached, which is the required capacity.
     */
    if (totalPassengers > minTram)
      minTram = totalPassengers; // Update minimum required tram capacity.
  }
  // Prints the calculated minimum required tram capacity, followed by a newline.
  printf("%d\n", minTram);
  // Standard program exit.
  return 0;