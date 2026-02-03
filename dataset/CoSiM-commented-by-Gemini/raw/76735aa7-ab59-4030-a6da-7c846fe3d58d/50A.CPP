/**
 * @file 50A.CPP
 * @brief Solution to the "Domino piling" problem.
 * @see https://codeforces.com/problemset/problem/50/A
 *
 * This program calculates the maximum number of dominoes that can be placed on
 * a rectangular board. Each domino covers two squares.
 *
 * Algorithm:
 * 1. Read the dimensions of the board (M and N).
 * 2. The total number of squares is M * N.
 * 3. Each domino covers 2 squares, so the maximum number of dominoes is
 *    (M * N) / 2.
 *
 * The provided code is overly complex for this problem. A simpler solution is
 * to read M and N and print (M * N) / 2. The existing code appears to be
 * solving a different, more complex problem related to counting digits.
 *
 * Time Complexity: O(1)
 * Space Complexity: O(1)
 */

#include <iostream>
#include <math.h>
#define ll long long
using namespace std;
int main() {
  // This solution appears to be for a problem other than 50A.
  // The logic calculates the total number of digits in all numbers up to n.
  ll int n;
  cin >> n;
  ll int a[13] = {0};
  // Block Logic: Initializes the array `a` where a[i] is intended to store
  // the count of numbers with exactly `i` digits.
  for (int i = 1; i < 13; i++) {
    a[i] = pow(10, i) - 1;
  }
  for (int i = 2; i < 13; i++) {
    for (int j = 1; j <= i - 1; j++) {
      a[i] = a[i] - a[j];
    }
  }

  ll int count = 0;
  // Invariant: Iterates through the number of digits from 1 to 12.
  for (int i = 1; i < 13; i++) {
    // Pre-condition: If `n` is greater than the count of numbers with `i`
    // digits, add the total digits for this group to the count.
    if (n - a[i] >= 0) {
      count += a[i] * i;
      n = n - a[i];
    } else {
      count += n * i;
      break;
    }
  }
  cout << count << endl;
}