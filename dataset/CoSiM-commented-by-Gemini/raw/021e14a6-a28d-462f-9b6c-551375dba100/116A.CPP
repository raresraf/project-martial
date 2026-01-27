/**
 * @file 116A.CPP
 * @brief Solution for Codeforces problem 116A "Tram".
 *
 * This program calculates the minimum capacity a tram needs to have to
 * accommodate all passengers during a series of stops, given the number
 * of passengers exiting and entering at each stop.
 *
 * Algorithm: Iterative simulation of passenger flow, tracking current
 * passengers and maximum capacity required.
 * Time Complexity: O(N) where N is the number of tram stops.
 * Space Complexity: O(1)
 */
#include <algorithm>
#include <bits/stdc++.h>
#include <cmath>
#include <string>
#define ll long long int
#define ull unsigned long long int
using namespace std;

int main() {
#ifndef ONLINE_JUDGE
  // Functional Utility: Redirects standard input from "input.txt" and standard output to "output.txt".
  // This block is typically used for local debugging in competitive programming.
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  int n;
  // Block Logic: Reads the total number of stops.
  cin >> n;
  // Functional Utility: Initializes `capa` (maximum capacity encountered so far) and `curr` (current passengers in tram).
  int capa = 0, curr = 0;
  int a, b;
  // Block Logic: Reads the passengers exiting (`a`) and entering (`b`) at the first stop.
  // Updates `curr` and `capa` based on the first stop's passenger change.
  cin >> a >> b;
  curr = b;
  capa = b;
  /**
   * Block Logic: Iterates through the remaining `n-1` stops to simulate passenger flow.
   * Pre-condition: `n` represents the total number of stops, and the first stop has been processed.
   * Invariant: `curr` always reflects the current number of passengers, and `capa` stores
   * the maximum number of passengers observed at any point, representing the minimum required capacity.
   */
  for (int i = 1; i < n; i++) {
    // Reads passengers exiting (`a`) and entering (`b`) at the current stop.
    cin >> a >> b;
    // Updates `curr` by subtracting exiting passengers and adding entering passengers.
    curr = curr + b - a;
    // Updates `capa` if `curr` exceeds the previously recorded maximum.
    capa = max(capa, curr);
  }
  // Functional Utility: Prints the calculated minimum required tram capacity.
  cout << capa << "\n";
  return 0;
}
