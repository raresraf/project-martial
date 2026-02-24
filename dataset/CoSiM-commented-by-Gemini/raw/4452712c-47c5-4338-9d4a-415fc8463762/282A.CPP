/**
 * @file raw/4452712c-47c5-4338-9d4a-415fc8463762/282A.CPP
 * @brief Solution to Codeforces problem 282A - Bit++.
 *
 * @algorithm
 * The problem involves a variable 'x' initialized to 0 and a series of
 * increment (e.g., "++X", "X++") or decrement (e.g., "--X", "X--")
 * operations. The goal is to find the final value of 'x'.
 *
 * The core insight is that any operation containing "++" increases the value
 * by one, and any operation containing "--" decreases it by one. The exact
 * form of the operation (prefix or postfix) is irrelevant to the final sum.
 *
 * This implementation simplifies the logic by checking only the middle character
 * of the operation string (at index 1). If this character is '+', it's an
 * increment; if it's '-', it's a decrement.
 *
 * The algorithm is as follows:
 * 1. Initialize a counter `count` to 0, representing the variable 'x'.
 * 2. Read `n`, the total number of operations.
 * 3. Loop `n` times to process each operation.
 * 4. In each iteration, read the operation as a string.
 * 5. Inspect the character at index 1 of the string.
 * 6. If `string[1]` is '+', increment `count`.
 * 7. If `string[1]` is '-', decrement `count`.
 * 8. After the loop, print the final value of `count`.
 *
 * @complexity
 * - Time Complexity: O(n), where n is the number of operations, because the
 *   loop runs `n` times, and each operation inside is O(1).
 * - Space Complexity: O(MAX), due to the large character array `x`. However,
 *   since each operation string is short, the practical space usage is much
 *   lower. A smaller, constant-size buffer would yield O(1) space complexity.
 */
#include <bits/stdc++.h>

// These macros and typedefs are defined but not all are used in this specific problem.
#define pi 2.0 * acos(0.0)
#define MAX 100005
typedef long long ll;
using namespace std;

int main() {
  // i: loop counter.
  // j, k: unused variables.
  // n: number of statements.
  // count: the value of the variable x, initialized to 0.
  int i, j, k, n, count = 0;
  // x: a character array to store the operation string.
  // Its size is much larger than needed for this problem.
  char x[MAX];

  // Read the number of operations.
  cin >> n;

  /**
   * @brief This loop processes each of the 'n' operations.
   */
  for (i = 0; i < n; i++) {
    // Read the operation string (e.g., "++X", "X--").
    cin >> x;
    // Check the middle character to determine if it's an increment or decrement.
    if (x[1] == '+')
      count = count + 1; // Increment operation.
    else if (x[1] == '-')
      count = count - 1; // Decrement operation.
  }

  // Print the final value of the counter.
  cout << count;
  return 0;
}
