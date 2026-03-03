/**
 * @file 670A.CPP
 * @brief Solution for Codeforces Problem 670A - "Holidays".
 * This program calculates the minimum and maximum possible number of holidays
 * within a given period of 'n' days, considering Saturday and Sunday as holidays.
 * The logic simulates two scenarios: one starting the period on a Monday (to find minimum holidays)
 * and another starting on a Saturday (to find maximum holidays).
 * Algorithm: Iterative simulation of days of the week.
 * Time Complexity: O(N), where N is the number of days, due to iterating through N days.
 * Space Complexity: O(1), as only a fixed-size array and a few variables are used.
 */
/*May The Force Be With Me*/
#include <bits/stdc++.h> // Includes most standard C++ libraries
#include <ctype.h>       // For character handling functions
#include <stdio.h>       // For standard input/output functions

#pragma GCC optimize("Ofast") // Compiler optimization directive
#define ll long long          // Defines 'll' as a shorthand for 'long long'
#define MOD 1000000007        // Defines a constant for modulo operations
#define endl "\n"             // Defines 'endl' to use newline character for faster output
#define vll vector<long long> // Defines 'vll' as a shorthand for vector of long longs
#define mll map<long long, long long> // Defines 'mll' as a shorthand for map with long long keys and values
#define pll pair<long long, long long> // Defines 'pll' as a shorthand for pair of long longs
#define all(c) c.begin(), c.end()     // Macro to get iterators to the beginning and end of a container
#define rall(c) c.rbegin(), c.rend()  // Macro to get reverse iterators to the beginning and end of a container
#define pb push_back                  // Defines 'pb' as a shorthand for push_back
#define f first                       // Defines 'f' as a shorthand for pair's first element
#define s second                      // Defines 's' as a shorthand for pair's second element
#define inf INT_MAX                   // Defines 'inf' as a shorthand for maximum integer value
#define size_1d 10000000              // Defines a constant for a 1D array size
#define size_2d 1000                  // Defines a constant for a 2D array size
// Snippets: delta, sieve, fastexp, dfs, bfs, dijkstra, floodfill

using namespace std; // Uses the standard namespace

ll n; // Global variable to store the number of days

/**
 * @brief Reads the input value for 'n' (number of days).
 * Functional Utility: Handles standard input for the problem's primary parameter.
 */
void Input() { cin >> n; }

/**
 * @brief Solves the "Holidays" problem by calculating minimum and maximum possible holidays.
 * This function simulates two scenarios to find the boundary conditions for holiday counts.
 */
void Solve() {
  // a[] represents the days of a week, where 0 is a workday and 1 is a holiday.
  // Assuming t=0 is Monday, then a[5] is Saturday and a[6] is Sunday.
  ll a[] = {0, 0, 0, 0, 0, 1, 1};
  
  // Variable 't' represents the current day of the week (0-indexed, Mon=0, ..., Sun=6).
  // Variable 'ans' accumulates the count of holidays for the current scenario.
  ll t = 0; // Starts simulation on Monday for minimum holidays.
  ll ans = 0;

  /**
   * Block Logic: Calculates the minimum number of holidays.
   * This loop simulates 'n' days starting from a Monday (t=0) to find the minimum
   * possible holidays.
   * Invariant: 'ans' counts holidays encountered so far, and 't' tracks the current day of the week.
   */
  for (ll i = 0; i < n; i++) {
    if (a[t]) // If the current day is a holiday
      ans++;   // Increment holiday count
    t++;     // Move to the next day
    t %= 7;  // Reset day of week counter if it exceeds 6 (Sunday)
  }
  // Prints the minimum number of holidays.
  cout << ans << " ";
  
  ans = 0;    // Reset holiday count for the next scenario
  t = 5;      // Starts simulation on Saturday for maximum holidays.

  /**
   * Block Logic: Calculates the maximum number of holidays.
   * This loop simulates 'n' days starting from a Saturday (t=5) to find the maximum
   * possible holidays.
   * Invariant: 'ans' counts holidays encountered so far, and 't' tracks the current day of the week.
   */
  for (ll i = 0; i < n; i++) {
    if (a[t]) // If the current day is a holiday
      ans++;   // Increment holiday count
    t++;     // Move to the next day
    t %= 7;  // Reset day of week counter if it exceeds 6 (Sunday)
  }
  // Prints the maximum number of holidays.
  cout << ans;
}

int main() {
  // Functional Utility: Optimizes C++ standard streams for competitive programming.
  // Disables synchronization with C's stdio and unties cin from cout, improving I/O performance.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  
  // Conditional compilation for local debugging vs. online submission.
#ifndef ONLINE_JUDGE
  // Functional Utility: Redirects standard input from "input.txt" and standard output to "output.txt"
  // for easier local testing without console interaction.
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  ll T = 1; // Variable for number of test cases, initialized to 1 as per typical competitive programming setup.
  // cin >> T; // Commented out, implying a single test case or custom handling.
  // ll t = 1; // Commented out, likely for test case numbering if multiple test cases were enabled.
  
  /**
   * Block Logic: Main loop to process test cases.
   * Invariant: The loop continues as long as there are test cases (T > 0).
   */
  while (T--) {
    Input(); // Calls function to read input for the current test case.
    // cout << "Case " << t << ": "; // Commented out, for printing test case number.
    Solve(); // Calls function to solve the problem for the current test case.
    // t++; // Commented out, for incrementing test case number.
  }
  return 0; // Indicates successful program execution.
}