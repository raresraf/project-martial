/**
 * @file 116A.CPP
 * @brief Calculates the minimum required capacity of a tram.
 *
 * This program simulates passenger flow on a tram. By tracking the number
 * of passengers at each stop, it determines the peak passenger count, which
 * corresponds to the minimum capacity the tram must have.
 *
 * @algorithm Greedy Simulation
 * @complexity
 * Time: O(N), where N is the number of tram stops, as the program iterates
 *       through the stops once.
 * Space: O(1), as it uses a fixed number of variables to store the state.
 */
#include <iostream>
using namespace std;

/**
 * @brief Main function to handle input and run the simulation.
 * @param argc Argument count (unused).
 * @param argv Argument vector (unused).
 */
int main(int argc, const char *argv[]) {
  // n: the number of stops to process.
  int n;
  cin >> n;

  // c: current number of passengers on the tram.
  // max: tracks the maximum number of passengers seen so far.
  // l: passengers leaving at a stop.
  // e: passengers entering at a stop.
  int c = 0, max = 0, l, e;
  
  // Pre-condition: Tram is empty, so c and max are 0.
  // Block Logic: Loop 'n' times, once for each stop.
  while (n--) {
    cin >> l >> e;
    // Update the current passenger count.
    c -= l;
    c += e;
    // If the current count is a new maximum, record it.
    if (c > max)
      max = c;
  }
  
  // Output the highest passenger count, which is the minimum required capacity.
  cout << max;
  
  return 0;
}
