/**
 * @file 116A.CPP
 * @brief Calculates the minimum required capacity of a tram.
 *
 * This program simulates the passenger flow on a tram over a series of stops.
 * It tracks the current number of passengers and finds the maximum value reached,
 * which represents the minimum capacity the tram must have.
 *
 * @algorithm Greedy Simulation
 * @complexity
 * Time: O(N), where N is the number of tram stops, because the program
 *       iterates through the stops a single time.
 * Space: O(1), as it only uses a few variables to maintain the current state.
 */
#include <iostream>
#include <algorithm> // For std::max
#include <vector>    // Though not used, often included with <bits/stdc++.h>

using namespace std;

// --- Common Competitive Programming Boilerplate ---
// These macros and typedefs are part of a template and are not all used in this specific solution.
#define length(a) (int)a.size()
#define all(a) a.begin(), a.end()
#define tolowercase(s) transform(s.begin(), s.end(), s.begin(), ::tolower);
#define touppercase(s) transform(s.begin(), s.end(), s.begin(), ::toupper);
#define lsb(x) (x & (-x))
#define ctoi(a) (int)a - 48
#define itoc(a) a + '0'
#define MOD 1000000007
#define ll long long

/**
 * @brief Main function to execute the tram simulation.
 */
int main() {
  // These lines disable synchronization with C stdio and untie cin/cout,
  // speeding up I/O operations, which is crucial in competitive programming.
  ios::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);
  
  // n: the total number of tram stops.
  int n;
  cin >> n;
  
  // highest: tracks the maximum number of passengers on the tram at any point.
  // curr: tracks the current number of passengers on the tram.
  int highest = 0;
  int curr = 0;
  
  // Pre-condition: Tram is empty, so curr and highest are 0.
  // Block Logic: Loop through each of the n stops.
  for (int i = 0; i < n; i++) {
    // a: passengers exiting, b: passengers entering.
    int a, b;
    cin >> a >> b;
    
    // Update the current number of passengers.
    curr -= a;
    curr += b;
    
    // Update the maximum capacity required if the current number is greater.
    highest = max(highest, curr);
  }
  
  // Output the minimum capacity required.
  cout << highest << "
";
  
  return 0;
}
