/**
 * @file 282A.CPP
 * @brief Solution to Codeforces problem 282A - Bit++.
 *
 * @algorithm
 * The problem involves a variable, `X`, initialized to 0. A sequence of
 * operations ("++X", "X++", "--X", "X--") is given, and the final value of `X`
 * must be determined.
 *
 * The core logic relies on a simple observation:
 * - Any statement containing "++" increments the value of X.
 * - Any statement containing "--" decrements the value of X.
 *
 * The code implements a clever shortcut. In all possible input strings
 * ("++X", "X++", "--X", "X--"), the character at index 1 is unique to the
 * operation type ('+' for increment, '-' for decrement).
 *
 * The algorithm is as follows:
 * 1. Initialize a counter to 0. This represents the value of `X`.
 * 2. Read the number of statements, `n`.
 * 3. Loop `n` times, reading each statement as a string.
 * 4. In each iteration, inspect the character at index 1 of the string.
 * 5. If it's '+', increment the counter.
 * 6. If it's '-', decrement the counter.
 * 7. After the loop, the counter holds the final value of `X`.
 *
 * @complexity
 * - Time Complexity: O(N), where N is the number of statements, as the loop
 *   runs N times with constant time operations inside.
 * - Space Complexity: O(S), where S is the maximum length of a statement string
 *   (which is a small constant, so effectively O(1)).
 */
#include <bits/stdc++.h>

// The following macros are defined but not used in this specific solution.
#define pi 2.0 * acos(0.0)
#define MAX 100005

typedef long long ll; // Defines 'll' as an alias for 'long long', not used here.
using namespace std;

int main() {
  // i: loop counter.
  // n: the number of statements to be processed.
  // count: represents the value of the variable 'X', initialized to 0.
  int i, j, k, n, count = 0; // j and k are declared but not used.
  
  // x: character array to store each input statement. MAX is 100005.
  char x[MAX];

  cin >> n;

  /**
   * Block Logic: Process each of the `n` statements.
   * Pre-condition: `n` holds the number of statements, `count` is 0.
   * Invariant: After `i` iterations, `count` reflects the result of the first `i` operations.
   */
  for (i = 0; i < n; i++) {
    cin >> x;
    // The logic hinges on checking the middle character of the 3-character string.
    // If it's '+', it's an increment operation.
    if (x[1] == '+')
      count = count + 1;
    // If it's '-', it's a decrement operation.
    else if (x[1] == '-')
      count = count - 1;
  }

  // Output the final value of the counter.
  cout << count;
  return 0;
}
