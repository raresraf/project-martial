/**
 * @file 282A.CPP
 * @brief Solution to the "Bit++" problem.
 * @details This program calculates the final value of a variable 'X' after a series of increment and decrement operations.
 * The operations are given as strings: "++X", "X++", "--X", and "X--". The initial value of X is 0.
 *
 * @section algorithm Algorithm
 * The program initializes a counter variable to zero. It then reads the number of operations `n`
 * and iterates `n` times. In each iteration, it reads an operation string and checks if it's an
 * increment or decrement operation using `strcmp`. The counter is updated accordingly. Finally,
 * the program prints the final value of the counter.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(n), where `n` is the number of operations, because the program iterates through all operations once.
 * - Space Complexity: O(1), as the program uses a fixed amount of memory regardless of the input size.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // 'init' holds the value of X, initialized to 0.
  int init = 0, n;
  char str[4];
  cin >> n;
  // Loop through all 'n' operations.
  for (int i = 0; i < n; i++) {
    cin >> str;
    // Check for pre-increment operation.
    if (!strcmp(str, "++X"))
      ++init;
    // Check for post-increment operation.
    if (!strcmp(str, "X++"))
      init++;

    // Check for pre-decrement operation.
    if (!strcmp(str, "--X"))
      --init;
    // Check for post-decrement operation.
    if (!strcmp(str, "X--"))
      init--;
  }
  // Print the final value of X.
  cout << init << endl;
  return 0;
}
