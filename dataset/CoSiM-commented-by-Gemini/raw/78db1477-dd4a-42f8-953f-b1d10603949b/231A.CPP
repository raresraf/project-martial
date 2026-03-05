/**
 * @file 231A.CPP
 * @brief Solves a puzzle involving a unary representation of an equation (e.g., ||+|||=|||||).
 * The program checks if an equation A+B=C is valid, or can be made valid by moving
 * a single stick.
 *
 * Algorithm:
 * 1. Parse the input string to determine the unary values of A, B, and C.
 * 2. Check if the equation A + B = C is already satisfied.
 * 3. If not, check the two primary single-move transformations:
 *    a. A + B = C - 2: This can be solved by moving one stick from C to either A or B.
 *       - (A+1) + B = C-1
 *       - A + (B+1) = C-1
 *    b. A + B = C + 2: This can be solved by moving one stick from either A or B to C.
 *       - (A-1) + B = C+1
 *       - A + (B-1) = C+1
 * 4. If a valid transformation is found, print the corrected unary equation. Otherwise, output "Impossible".
 *
 * Time Complexity: O(L) per test case, where L is the length of the input string.
 * Space Complexity: O(L) to store the input string.
 */
#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <math.h>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <time.h>
#include <vector>
#define mset(a, val) memset(a, val, sizeof(a))
#define up(i, s, t) for (int i = (s); i < (t); i += 1)
#define down(i, s, t) for (int i = (s); i > (t); i -= 1)
#define rd1(a) scanf("%d", &a)
#define rd2(a, b) scanf("%d %d", &a, &b)
#define rd3(a, b, c) scanf("%d %d %d", &a, &b, &c)
#define rd4(a, b, c, d) scanf("%d %d %d %d", &a, &b, &c, &d)
#define pii pair<int, int>

typedef long long ll;
using namespace std;
const int MAXINT = 1 << 31 - 1;
const ll MAXLONG = (ll)1 << 63 - 1;

/**
 * @brief Prints a specified number of vertical bars '|' to represent a unary number.
 * @param a The integer value to be represented.
 */
void sti(int a) {
  up(i, 0, a) { printf("|"); }
}

/**
 * @brief Outputs the unary equation in the format A+B=C.
 * @param a The integer value of the first operand.
 * @param b The integer value of the second operand.
 * @param c The integer value of the sum.
 */
void output(int a, int b, int c) {
  sti(a);
  printf("+");
  sti(b);
  printf("=");
  sti(c);
  cout << endl;
}

/**
 * @brief Checks if a number is greater than zero.
 * In the context of the problem, this ensures a unary number pile is not empty.
 * @param a The integer to check.
 * @return True if a > 0, false otherwise.
 */
bool check_one(int a) {
  if (0 < a)
    return true;
  return false;
}

/**
 * @brief Checks if all three numbers are greater than zero.
 * This is a precondition for a valid equation state, as no number can be zero.
 * @param a The first number.
 * @param b The second number.
 * @param c The third number.
 * @return True if all numbers are positive, false otherwise.
 */
bool check(int a, int b, int c) {
  if (check_one(a) && check_one(b) && check_one(c)) {
    return true;
  }
  return false;
}

int main() {
/**
 * @brief Redirects standard input and output to files for local testing.
 * This block is ignored when the code is compiled in an online judge environment.
 */
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  string str;
  /**
   * @brief Main processing loop that reads and solves one equation per line.
   */
  while (cin >> str) {
    /**
     * @brief Parses the input string to count the number of sticks for each operand (a, b) and the result (c).
     */
    int plus = str.find('+');
    int equa = str.find('=');

    int a = plus;
    int b = equa - plus - 1;
    int c = str.size() - equa - 1;

    /**
     * @brief This block checks for a valid equation state.
     * It first checks if the equation is already correct (a + b == c).
     * If not, it systematically evaluates all possible single-stick moves.
     */
    if (check(a, b, c) && a + b == c) {
      // Case 1: The equation is already correct.
      output(a, b, c);
    } else if (check(a + 1, b, c - 1) && a + b + 1 == c - 1) {
      // Case 2: Move one stick from C to A.
      // This is valid if a + b = c - 2, and C has a stick to give (c > 1).
      output(a + 1, b, c - 1);
    } else if (check(a, b + 1, c - 1) && a + b + 1 == c - 1) {
      // Case 3: Move one stick from C to B.
      // This is valid if a + b = c - 2, and C has a stick to give (c > 1).
      output(a, b + 1, c - 1);
    } else if (check(a - 1, b, c + 1) && a + b - 1 == c + 1) {
      // Case 4: Move one stick from A to C.
      // This is valid if a + b = c + 2, and A has a stick to give (a > 1).
      output(a - 1, b, c + 1);
    } else if (check(a, b - 1, c + 1) && a + b - 1 == c + 1) {
      // Case 5: Move one stick from B to C.
      // This is valid if a + b = c + 2, and B has a stick to give (b > 1).
      output(a, b - 1, c + 1);
    } else {
      // If no single-stick move can correct the equation.
      puts("Impossible");
    }
  }
  return 0;
}