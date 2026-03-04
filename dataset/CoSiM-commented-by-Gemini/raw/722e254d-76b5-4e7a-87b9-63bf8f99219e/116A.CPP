/**
 * @file tram.cpp
 * @brief Solution for Codeforces Problem 116A - "Tram".
 *
 * This program calculates the minimum required capacity for a tram by simulating
 * passenger flow over a series of stops. It tracks the current number of
 * passengers and identifies the maximum occupancy at any point during the journey.
 *
 * @algorithm Simulation with running total and maximum tracking.
 * @time_complexity O(N), where N is the number of stops, as it processes each
 *                  stop in a single pass.
 * @space_complexity O(1), using a constant number of variables for state.
 */
/*
 * tram.cpp
 * http://codeforces.com/problemset/problem/116/A
 *  Created on: Oct 26, 2017
 *      Author: athena
 * 31757462 	GNU C++14 	Accepted 	30 ms 	8 KB 	2017-10-26
 * 18:52:15 	2017-10-26 18:52:16
 */

#include <iostream>
using namespace std;

/**
 * @brief Main function to calculate the minimum tram capacity.
 */
int main() {
  int n;
  cin >> n;

  int a, b;

  int counter = 0;
  int max = 0;

  /**
   * @brief Simulates the passenger flow at each stop.
   * This loop iterates 'n' times. In each iteration, it reads the number of
   * passengers exiting ('a') and entering ('b'), updates the current passenger
   * count ('counter'), and adjusts the maximum capacity ('max') if the current
   * count is a new high.
   */
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    counter -= a;
    counter += b;
    if (counter > max) {
      max = counter;
    }
  }

  // Output the calculated minimum capacity.
  cout << max << endl;

  return 0;
}