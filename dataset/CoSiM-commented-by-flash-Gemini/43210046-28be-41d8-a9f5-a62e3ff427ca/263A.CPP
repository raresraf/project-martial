/**
 * @file 263A.CPP
 * @brief Solution to a competitive programming problem (Codeforces 263A - "Beautiful Matrix").
 *
 * This program determines the minimum number of moves required to bring the
 * single '1' in a 5x5 matrix to its central position (row 2, column 2, 0-indexed).
 * A move consists of swapping two adjacent rows or two adjacent columns.
 * This is equivalent to calculating the Manhattan distance from the '1' to the center.
 *
 * Algorithm: Matrix traversal and Manhattan distance calculation.
 * It iterates through the 5x5 matrix to find the coordinates (i, j) of the
 * cell containing the value '1'. Once found, it calculates the Manhattan distance
 * as `abs(2 - i) + abs(2 - j)`.
 *
 * Time Complexity: O(R*C), where R is the number of rows and C is the number of columns.
 * Since R=5 and C=5, this is effectively O(1).
 * Space Complexity: O(1) as the matrix size is fixed and small, and only a few variables are used.
 */
#include <algorithm> // Required for std::abs (indirectly through bits/stdc++.h or explicitly if used).
#include <assert.h>  // For debugging, asserts conditions.
#include <bitset>    // For bit manipulation.
#include <ctype.h>   // For character type functions (e.g., isdigit, toupper).
#include <deque>     // For double-ended queues.
#include <iostream>  // Required for input/output operations (cin, cout).
#include <map>       // For associative containers.
#include <math.h>    // For mathematical functions (e.g., abs, pow, sqrt).
#include <queue>     // For queue data structure.
#include <set>       // For set data structure.
#include <stack>     // For stack data structure.
#include <stdio.h>   // For C-style input/output (scanf, printf).
#include <stdlib.h>  // For general utilities (e.g., abs, malloc).
#include <string.h>  // For C-style string manipulation (e.g., strlen, strcpy).
#include <string>    // For std::string.
#include <time.h>    // For time-related functions.
#include <vector>    // For std::vector.

using namespace std;

// Competitive programming macros for shorthand.
#define sp system("pause") // Pauses console execution (Windows specific, generally avoided in competitive programming).
#define FOR(i, a, b) for (int i = a; i <= b; ++i) // Loop from a to b (inclusive).
#define FORD(i, a, b) for (int i = a; i >= b; --i) // Loop from a to b (inclusive, reverse).
#define REP(i, n) FOR(i, 0, (int)n - 1)          // Loop from 0 to n-1.
#define pb(x) push_back(x)                       // Functional Utility: Appends an element to a vector.
#define mp(a, b) make_pair(a, b)                 // Functional Utility: Creates a pair.
#define MS0(x) memset(x, 0, sizeof(x))           // Sets all bytes of a memory block to 0.
#define MS1(x) memset(x, 1, sizeof(x))           // Sets all bytes of a memory block to 1.
#define SORT(a, n) sort(begin(a), begin(a) + n)  // Sorts an array.
#define REV(a, n) reverse(begin(a), begin(a) + n) // Reverses an array.
#define ll long long                             // Type alias for long long.
#define pii pair<int, int>                       // Type alias for pair of integers.
#define MOD 1000000007                           // Modulo constant.

// Global declaration of a 2D array, though not used in the main logic here.
int a[6][6]; // Declares a 6x6 integer array (likely for problem-specific context or boilerplate).

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads a 5x5 matrix, finds the position of '1', and calculates the Manhattan distance
 * to move it to the center (2,2).
 */
int main() {

  int num; // Variable to store the integer read from input.
  // Block Logic: Iterate through a 5x5 grid (0-indexed rows 0-4, columns 0-4).
  // Invariant: 'i' represents the current row index, 'j' represents the current column index.
  REP(i, 5) { // Loop for rows.
    REP(j, 5) { // Loop for columns.
      cin >> num; // Pre-condition: Read an integer for the current cell.
      // Block Logic: Check if the current number is '1'.
      if (num == 1) {
        // Functional Utility: Calculate and print the Manhattan distance.
        // The center of a 0-indexed 5x5 matrix is (2,2).
        // abs(2 - i) is row moves, abs(2 - j) is column moves.
        cout << abs(2 - i) + abs(2 - j) << endl; 
      }
    }
  }
  return 0; // Indicate successful execution.
}
