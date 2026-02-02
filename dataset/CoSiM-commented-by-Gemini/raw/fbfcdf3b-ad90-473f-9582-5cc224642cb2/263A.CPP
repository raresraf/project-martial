#include <bits/stdc++.h>
#include <iostream>
using namespace std;
/**
 * @brief Calculates the Manhattan distance of a '1' in a 5x5 matrix to the center.
 *
 * This program reads a 5x5 matrix of integers and finds the location of the
 * integer '1'. It then calculates the Manhattan distance (number of moves
 * horizontally and vertically) from this location to the center of the matrix,
 * which is at position (3, 3).
 *
 * @return 0 on successful execution.
 */
int main() {
  int i, j, p, q, n;
  for (i = 1; i <= 5; i++) {
    for (j = 1; j <= 5; j++) {
      cin >> n;
      if (n == 1) {
        p = i;
        q = j;
      }
    }
  }
  if (p > 3)
    p = p - 3;
  else
    p = 3 - p;
  if (q > 3)
    q = q - 3;
  else
    q = 3 - q;
  p = p + q;
  cout << p;
  return 0;
}