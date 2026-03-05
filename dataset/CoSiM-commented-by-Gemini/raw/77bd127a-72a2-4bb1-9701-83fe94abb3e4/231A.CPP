/**
 * @file 231A.CPP
 * @brief An attempted solution to the "Team" problem with several implementation flaws.
 *
 * The program intends to calculate the number of problems a three-person team will solve,
 * where a problem is solved if at least two members are sure.
 *
 * NOTE: This code is fundamentally broken and will not compile or run correctly due to
 * multiple issues. It is provided here for analysis of its intended logic versus its
 * flawed implementation.
 *
 * @intended_algorithm
 * 1. Read the number of problems, `n`.
 * 2. Store the opinions (0 or 1) for all `n` problems and 3 members in a 2D array.
 * 3. Iterate through the stored opinions for each problem.
 * 4. If the sum of opinions for a problem is 2 or greater, increment a counter.
 * 5. Print the final counter.
 *
 * @implementation_flaws
 * 1.  **Broken Macro:** `#define long long ll;` includes a semicolon, which will cause
 *     compilation errors wherever `ll` might be used.
 * 2.  **Global Fixed-Size Array:** The code uses a global array `arr[1000][3]`, which limits
 *     the number of problems to 1000 and can lead to a buffer overflow if `n > 1000`.
 * 3.  **Incorrect Loop:** The loop `for (int *i : arr)` in `solve()` is not standard C++ for
 *     iterating a 2D C-style array. On compilers that might accept it (like GCC as an
 *     extension), it will iterate 1000 times (the declared size of `arr`), not `n` times.
 *     This means it will process uninitialized data if `n < 1000`, leading to an
 *     incorrect answer.
 *
 * @complexity
 * Intended Time Complexity: O(n)
 * Intended Space Complexity: O(n)
 */

#include <iostream>

using namespace std;

// BUG: This macro is syntactically incorrect due to the trailing semicolon.
// It would cause a compile error if `ll` were used.
#define long long ll;

// Global variables are used, which is generally poor practice.
// BUG: `arr` is a fixed-size array, which can lead to buffer overflows if n > 1000.
int n, arr[1000][3];

void solve() {
  int answer = 0;
  
  // BUG: This range-based for loop syntax is incorrect for a raw 2D array.
  // It will iterate 1000 times (the full size of `arr`), not `n` times,
  // leading to incorrect results by processing uninitialized memory.
  // A correct loop would be `for (int i = 0; i < n; i++)`.
  for (int *i : arr) {
      // Inside the loop, `i` would be a pointer to a row `int[3]`.
      // The logic `i[0] + i[1] + i[2]` is correct for a single row.
    if (i[0] + i[1] + i[2] > 1)
      answer++;
  }
  cout << answer;
}

int main() {
  cin >> n;
  // This loop correctly reads the input into the first `n` rows of the array.
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> arr[i][j];
    }
  }
  solve();
  // The return 0 is implicit.
}
