/**
 * @5dc555ef-736f-424e-b4ad-abc42e4520ef/236A.CPP
 * @brief Solution for a competitive programming problem involving generating pairs of numbers.
 *
 * This program generates a specific number of unique pairs (i, j) such that 1 <= i < j <= n.
 * The total number of pairs to be generated is determined by the formula 2*n + p.
 * This is a common pattern in problems where graph edges or interactions need to be listed.
 *
 * Algorithm:
 * 1. Read the number of test cases (cas).
 * 2. For each test case:
 *    a. Read integers 'n' and 'p'.
 *    b. Calculate the target number of pairs 'm' as 2 * n + p.
 *    c. Use nested loops: the outer loop iterates 'i' from 1 to 'n', and the inner loop
 *       iterates 'j' from 'i + 1' to 'n'. This ensures that 'i < j' and each pair is unique.
 *    d. Inside the inner loop, print the pair (i, j) and decrement 'm'.
 *    e. The loops terminate when 'm' becomes 0, indicating that the required number of pairs has been printed.
 *
 * Time Complexity: O(N^2) in the worst case per test case, as it iterates through pairs of 'n' elements.
 *                  Specifically, it's O(min(N^2, 2N+P)) since it stops after printing 'm' pairs.
 * Space Complexity: O(1) as it uses a fixed amount of memory for variables.
 */

#include <algorithm> // Functional Utility: Provides generic algorithms, not directly used in this code.
#include <cstdio>    // Functional Utility: Provides C-style input/output functions (scanf, printf).
#include <cstring>   // Functional Utility: Provides C-style string functions, not used in this code.
// #include <iostream> // Functional Utility: Not directly used, as cstdio is preferred for competitive programming with scanf/printf.

using namespace std;

int n, p; // Functional Utility: Global variables for n and p, though local scope in main would also work.

int main() {
  int i, j, cas;
  // Functional Utility: Reads the number of test cases.
  scanf("%d", &cas);
  // Block Logic: Loops through each test case.
  // Pre-condition: 'cas' is the number of test cases to process.
  // Invariant: The code inside the loop will execute 'cas' times.
  while (cas--) {
    // Functional Utility: Reads integers 'n' and 'p' for the current test case.
    scanf("%d%d", &n, &p);
    // Functional Utility: Calculates the target number of pairs to print.
    // 'm' represents the remaining count of pairs to be printed.
    int m = 2 * n + p;
    // Block Logic: Outer loop iterates for the first element 'i' of a pair.
    // Pre-condition: 'i' starts from 1.
    // Invariant: 'i' will go up to 'n' or until 'm' pairs are printed.
    for (i = 1; i <= n && m; i++)
      // Block Logic: Inner loop iterates for the second element 'j' of a pair.
      // Pre-condition: 'j' starts from 'i + 1' to ensure i < j.
      // Invariant: 'j' will go up to 'n' or until 'm' pairs are printed.
      for (j = i + 1; j <= n && m; j++)
        // Functional Utility: Prints the pair (i, j).
        // Inline: Decrements 'm' after printing each pair.
        printf("%d %d\n", i, j), m--;
  }
  return 0; // Functional Utility: Indicates successful program execution.
}
