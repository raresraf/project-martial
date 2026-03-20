/**
 * @file 231A.CPP
 * @brief Solution to Codeforces problem 231A - Sticks.
 *        This program attempts to solve a puzzle where an equation represented by sticks
 *        (e.g., "|||+|=") needs to be made valid by moving exactly one stick.
 *        The program analyzes the initial stick counts on both sides of the '+' and '='
 *        and attempts various single-stick movements to achieve a valid equation.
 * Algorithm:
 *   1. Read the input string representing the stick equation (e.g., "|||+|=").
 *   2. Parse the string to determine the initial number of sticks in the first operand ('a'),
 *      the second operand ('b'), and the result ('c').
 *      'a' is the count of sticks before '+', 'b' is the count between '+' and '=',
 *      'c' is the count after '='.
 *   3. Check several scenarios by moving one stick:
 *      - Moving a stick from 'c' to 'a' (a+1, b, c-1).
 *      - Moving a stick from 'c' to 'b' (a, b+1, c-1).
 *      - Moving a stick from 'a' to 'c' (a-1, b, c+1).
 *      - Moving a stick from 'b' to 'c' (a, b-1, c+1).
 *      - The original configuration (a, b, c) without moving any sticks if it's already valid.
 *   4. For each scenario, check if the stick counts are positive and if the equation `a + b == c` holds.
 *   5. If a valid configuration is found, print it; otherwise, print "Impossible".
 * Time Complexity: O(L) for parsing the string, where L is the length of the string.
 *                  The checking part is O(1) as it involves a fixed number of operations.
 * Space Complexity: O(L) for storing the input string.
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

void sti(int a) {
  /**
   * @brief Prints 'a' vertical bars ('|') to standard output.
   *        This function visually represents a number using sticks.
   * @param a The number of vertical bars to print.
   */
  up(i, 0, a) { printf("|"); }
}

void output(int a, int b, int c) {
  /**
   * @brief Prints the stick equation in the format "|||+|=|||".
   *        It uses the `sti` function to represent numbers as sticks.
   * @param a The number of sticks for the first operand.
   * @param b The number of sticks for the second operand.
   * @param c The number of sticks for the result.
   */
  sti(a);
  printf("+");
  sti(b);
  printf("=");
  sti(c);
  cout << endl;
}

bool check_one(int a) {
  /**
   * @brief Checks if a given integer 'a' is positive.
   * @param a The integer to check.
   * @return bool True if 'a' is greater than 0, False otherwise.
   */
  if (0 < a)
    return true;
  return false;
}

bool check(int a, int b, int c) {
  /**
   * @brief Checks if all three given integers are positive.
   *        This is used to ensure that the number of sticks in each part of the equation
   *        (operand1, operand2, result) remains valid (i.e., at least one stick).
   * @param a The first integer (number of sticks).
   * @param b The second integer (number of sticks).
   * @param c The third integer (number of sticks).
   * @return bool True if all three integers are greater than 0, False otherwise.
   */
  if (check_one(a) && check_one(b) && check_one(c)) {
    return true;
  }
  return false;
}

int main() {
  /**
   * @brief Main function implementing the solution for the "Sticks" problem.
   *
   * Reads an equation represented by sticks, parses the stick counts, and then
   * systematically attempts to make the equation valid by simulating the movement
   * of exactly one stick. If a valid configuration is found, it prints the
   * corrected equation; otherwise, it reports "Impossible".
   * @return int Returns 0 upon successful execution.
   */
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  string str;
  // Block Logic: Continuously reads equations from input until EOF.
  // Invariant: 'str' contains the current stick equation.
  while (cin >> str) {
    // Block Logic: Finds the positions of '+' and '=' operators to delineate stick groups.
    // Pre-condition: 'str' is a valid stick equation string.
    // Post-condition: 'plus' and 'equa' store the index of '+' and '=' respectively.
    int plus = str.find('+');
    int equa = str.find('=');

    // Block Logic: Calculates the initial number of sticks in each part of the equation.
    // 'a' = sticks before '+', 'b' = sticks between '+' and '=', 'c' = sticks after '='.
    int a = plus;
    int b = equa - plus - 1;
    int c = str.size() - equa - 1;

    // Block Logic: Checks various scenarios by moving one stick to make the equation valid.
    // Each 'if-else if' block represents a potential single-stick movement and checks
    // if the resulting stick counts are valid (positive) and satisfy the equation a + b = c.
    // Pre-condition: a, b, c are initial stick counts.
    // Post-condition: Prints a valid equation if found, or "Impossible".
    if (check(a, b, c) && a + b == c) {
      output(a, b, c);
    } else if (check(a + 1, b, c - 1) && a + b + 1 == c - 1) {
      output(a + 1, b, c - 1);
    } else if (check(a, b + 1, c - 1) && a + b + 1 == c - 1) {
      output(a, b + 1, c - 1);
    } else if (check(a - 1, b, c + 1) && a + b - 1 == c + 1) {
      output(a - 1, b, c + 1);
    } else if (check(a, b - 1, c + 1) && a + b - 1 == c + 1) {
      output(a, b - 1, c + 1);
    } else {
      puts("Impossible");
    }
  }
  return 0;
}