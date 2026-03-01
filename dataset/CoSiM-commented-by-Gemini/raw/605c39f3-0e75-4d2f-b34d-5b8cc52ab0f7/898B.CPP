/**
 * @file 898B.CPP
 * @brief Solves a linear Diophantine equation of the form ax + by = n.
 *
 * @problem Given three integers n, a, and b, the program seeks to find any pair of
 * non-negative integers (x, y) that satisfy the equation a*x + b*y = n.
 *
 * @algorithm The approach is a brute-force search over one of the variables. The code
 * iterates through all possible non-negative values for x (aliased as 'p' in the code)
 * such that `a*p <= n`. For each value of 'p', it checks if the remaining
 * part, `n - a*p`, is a non-negative multiple of 'b'. If it is, a valid pair (x, y)
 * has been found, and the loop terminates.
 *
 * @time_complexity O(n/a). The loop runs from p=0 up to n/a.
 * @space_complexity O(1).
 */
#include <iostream>
#include <vector>

int main() {

  long n, a, b;
  std::cin >> n >> a >> b;

  // Initialize solution coordinates to -1 to indicate 'not found'.
  long x(-1), y(-1);

  // Iterate through all possible values of the first coefficient 'x' (aliased as p).
  // Pre-condition: p must be non-negative and p*a must not exceed n,
  // since b*y must also be non-negative.
  for (long p = 0; p * a <= n; p++) {
    // Check if the remainder (n - p*a) is perfectly divisible by b.
    if ((n - p * a) % b == 0) {
      // A solution is found.
      x = p;
      y = (n - p * a) / b;
      // The first solution found is sufficient, so we can exit the loop.
      break;
    }
  }

  // If x is no longer -1, a solution was found.
  if (x >= 0) {
    printf("YES\n%ld %ld\n", x, y);
  } else {
    puts("NO");
  }

  return 0;
}
