/**
 * @file 339A.CPP
 * @brief Solution to a competitive programming problem (Codeforces 339A - "Helpful Maths").
 *
 * This program sorts an arithmetic expression consisting of numbers 1, 2, and 3
 * separated by '+' signs, arranging them in non-decreasing order.
 * This implementation uses a blend of C-style I/O and C++ standard library features.
 *
 * Algorithm: C-style input parsing, array storage, standard sorting, and C-style output.
 * It reads individual digits (as numbers) and the '+' characters. The digits are
 * stored in a fixed-size `long long int` array, `std::sort` is used to sort these
 * numbers, and finally, `printf` is used to output the sorted numbers separated by '+'.
 *
 * Time Complexity: O(N log N), where N is the number of digits/numbers in the input.
 * The input parsing is linear in the string length (L), and sorting N elements takes O(N log N).
 * Since N is roughly L/2, the overall complexity is dominated by sorting.
 * Space Complexity: O(N) for storing the `ch` array.
 */
/* ===================================*/
/* Author :   Moshiur Rahman Mohim    */
/* Email  :   mtmohim74@gmail.com     */
/* Institute: HSTU                    */
/* ===================================*/

#include <cctype>     // For character handling functions.
#include <cmath>      // For mathematical functions.
#include <cstdio>     // For C-style input/output (scanf, printf).
#include <cstdlib>    // For general utilities.
#include <cstring>    // For C-style string manipulation (strlen, memset).
#include <iostream>   // For C++ standard input/output.

#include <algorithm> // Required for std::sort.
#include <list>      // For std::list.
#include <map>       // For std::map.
#include <queue>     // For std::queue.
#include <set>       // For std::set.
#include <sstream>   // For string streams.
#include <stack>     // For std::stack.
#include <string>    // For std::string.
#include <vector>    // For std::vector.
using namespace std;

// Competitive programming macros and typedefs.
#define pi acos(-1.0) // Mathematical constant Pi.
#define mod 1000000007 // Modulo constant.
#define MAX 10000      // Maximum value constant.
#define FOR(i, b, n) for (LL i = b; i <= n; i++) // Loop from b to n (inclusive).
#define FORR(i, n, b) for (LL i = n; i >= b; i--) // Loop from n to b (inclusive, reverse).
// Library.
#define mem(a, b) memset(a, b, sizeof(a)) // Functional Utility: Sets memory to a value.
#define Sort(x) sort(x.begin(), x.end()) // Functional Utility: Sorts a vector.
#define Reverse(x) reverse(x.begin(), x.end()) // Functional Utility: Reverses a vector.
#define gcd(a, b) __gcd(a, b) // Functional Utility: Calculates GCD.
#define lcm(a, b) (a * (b / gcd(a, b))) // Functional Utility: Calculates LCM.
#define sq(x) (x) * (x) // Functional Utility: Squares a value.
// File.
#define READ(f) freopen(f, "r", stdin) // Functional Utility: Redirects stdin to a file.
#define WRITE(f) freopen(f, "w", stdout) // Functional Utility: Redirects stdout to a file.
// i/o.
#define sf scanf // Alias for scanf.
#define pf printf // Alias for printf.
#define pr1(x) cout << x << "\n" // Prints x followed by newline.
#define pr2(x, y) cout << x << " " << y << "\n" // Prints x, y followed by newline.
#define pr3(x, y, z) cout << x << " " << y << " " << z << "\n" // Prints x, y, z followed by newline.
// Test Case.
#define TEST(i, t)                                                             \
  LL i, t;                                                                     \
  scanf("%I64d", &t); /*Functional Utility: Reads number of test cases, loop for them.*/ \
  for (i = 1; i <= t; i++)

// Type definitions for various integer and floating-point types.
typedef unsigned int U;
typedef long int L;
typedef unsigned long int LU;
typedef long long int LL;
typedef unsigned long long int LLU;
typedef float F;
typedef double LF;
typedef char C;

/**
 * @brief Computes (B^P) % M using binary exponentiation (modular exponentiation).
 * @param B Base.
 * @param P Exponent.
 * @param M Modulo.
 * @return int The result of (B^P) % M.
 */
int BigMod(LL B, LL P, LL M) {
  LL R = 1;
  while (P > 0) {
    if (P % 2 == 1)
      R = (R * B) % M;
    P /= 2;
    B = (B * B) % M;
  }
  return (int)R;
}

/*............End............*/
/*.......Global Declaration.......*/
C c; // Global character variable, used to capture delimiter.

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads an arithmetic expression from standard input, extracts the numbers,
 * sorts them, and prints the sorted expression with '+' separators.
 */
int main() {
  // READ("input.txt"); // Commented out: file input redirection.
  // WRITE("output.txt"); // Commented out: file output redirection.
  LL ch[100], n; // 'ch' array to store numbers (max 100), 'n' for count of numbers.
  // Block Logic: Reads numbers and delimiters until a newline is encountered.
  // Functional Utility: scanf("%I64d%c", &ch[i], &c) reads a long long and then a character.
  // The character 'c' captures either '+' or '\n'.
  FOR(i, 0, 100) { // Loop to read up to 100 numbers.
    sf("%I64d%c", &ch[i], &c); // Read a number and the character that follows it.
    if (c == '\n') {          // If the character is a newline, it marks the end of the input.
      n = i + 1;              // 'n' is the count of numbers read.
      break;                  // Exit the loop.
    }
  }
  // Functional Utility: Sort the array of numbers in non-decreasing order.
  sort(ch, ch + n);
  // Block Logic: Print the sorted numbers with '+' separators.
  FOR(i, 0, n - 1) {
    pf("%I64d", ch[i]); // Print the current number.
    if (i != n - 1)     // If it's not the last number, print a '+'.
      pf("+");
  }
  cout << endl; // Add a newline at the end of the output.
  return 0;     // Indicate successful execution.
}
