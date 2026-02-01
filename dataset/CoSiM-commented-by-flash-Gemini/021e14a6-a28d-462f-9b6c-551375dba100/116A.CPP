/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem (Codeforces 116A).
 * This program simulates passenger movement on a tram to determine the maximum number of passengers
 * present on the tram at any single moment throughout its journey. The tram has 'n' stops,
 * and at each stop, a certain number of passengers exit and a certain number enter.
 * Algorithm: Iterative simulation. The program maintains a `current` passenger count and
 * a `maximum capacity` observed so far. For each tram stop, it updates the `current` count
 * by subtracting passengers who exit and adding passengers who enter. The `maximum capacity`
 * is then updated if the `current` count surpasses the previously recorded maximum.
 * Time Complexity: O(N), where N is the number of tram stops, as the program processes each stop once.
 * Space Complexity: O(1), as it only requires a few integer variables to store counts.
 */
// By archit-dev, contest: Codeforces Beta Round #87 (Div. 2 Only), problem: (A)
// Tram, Accepted, #, Copy

#include <algorithm>
#include <bits/stdc++.h>
#include <cmath>
#include <string>
#define ll long long int
#define ull unsigned long long int
using namespace std;

int main() {
// Block Logic: Conditional compilation for redirecting standard input/output.
// This is a common practice in competitive programming for local testing, allowing
// input to be read from "input.txt" and output to be written to "output.txt".
// Invariant: This block is active only when not compiling for an online judge.
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  int n;
  // Functional Utility: Reads the number of tram stops.
  cin >> n;
  int capa = 0, curr = 0; // Inline: 'capa' tracks maximum capacity, 'curr' tracks current passengers.
  int a, b;
  // Functional Utility: Reads the passengers exiting ('a') and entering ('b') at the first stop.
  cin >> a >> b;
  // Functional Utility: Initializes current passengers and maximum capacity with the passengers entering at the first stop.
  // Precondition: The tram starts empty, so 'a' for the first stop is effectively 0.
  curr = b;
  capa = b;
  // Block Logic: Iterates through the remaining tram stops (from the second stop onwards).
  // Invariant: 'curr' accurately reflects the number of passengers on the tram after processing each stop.
  for (int i = 1; i < n; i++) {
    // Functional Utility: Reads passengers exiting ('a') and entering ('b') for the current stop.
    cin >> a >> b;
    // Functional Utility: Updates the current number of passengers on the tram.
    // Logic: Current passengers = (previous current passengers - exited passengers + entered passengers).
    curr = curr + b - a;
    // Functional Utility: Updates the maximum capacity observed so far.
    // Logic: 'capa' stores the highest 'curr' value encountered.
    capa = max(capa, curr);
  }
  // Functional Utility: Prints the maximum capacity observed throughout the tram's journey, followed by a newline.
  cout << capa << "\n";
  return 0;
}
