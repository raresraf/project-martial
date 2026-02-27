/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem from Codeforces.
 * @details The program calculates the minimum required capacity for a tram by
 * finding the maximum number of passengers on board at any given time.
 *
 * @algorithm
 * The solution simulates the journey stop by stop.
 * 1. Read the number of stops, `a`.
 * 2. Initialize `sum` (current passengers) and `count` (maximum passengers) to 0.
 * 3. Loop `a` times, once for each stop.
 *    a. Read the number of passengers exiting (`down`) and entering (`up`).
 *    b. Calculate the net change in passengers for the stop.
 *    c. Add this net change to the `sum` to get the new current passenger count.
 *    d. If the new `sum` is greater than the max `count` seen so far, update `count`.
 * 4. After all stops, `count` holds the peak passenger number, which is the
 *    minimum required capacity. Print `count`.
 *
 * @complexity
 * Time Complexity: O(n), where n is the number of stops.
 * Space Complexity: O(1).
 */
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <vector>
using namespace std;
int main() {
  int up, down; // Passengers entering and exiting at a stop.
  int count = 0; // Tracks the maximum number of passengers seen (the answer).
  int i, j; // Loop counters, j is unused.
  int a; // The number of stops.
  int sum = 0; // Tracks the current number of passengers.
  
  cin >> a;
  
  /**
   * @block
   * @brief Simulate the passenger flow at each of the 'a' stops.
   */
  for (i = 1; i <= a; i++) {
    cin >> down >> up;
    // The 'up' variable is reused to store the net change in passengers.
    up = up - down;
    // Update the current passenger count with the net change.
    sum = sum + up;
    
    // If the current number of passengers is the new maximum, update the max counter.
    if (sum > count) {
      count = sum;
    }
  }
  
  cout << count << endl;

  return 0;
}
