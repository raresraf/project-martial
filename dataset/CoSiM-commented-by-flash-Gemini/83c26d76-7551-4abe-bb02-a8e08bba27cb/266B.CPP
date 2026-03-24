/**
 * @file 266B.CPP
 * @brief Solution to the "Queue at the School" competitive programming problem.
 *
 * This program simulates a queue management process over several time steps.
 * The queue consists of boys ('B') and girls ('G'). In each time step, if a boy
 * is immediately followed by a girl (i.e., "BG"), they swap positions to become "GB".
 * A key constraint is that each boy can only move one position forward per time step.
 * The program determines the final arrangement of the queue after `t` time steps.
 *
 * Algorithm: Iterative simulation with local swaps.
 * Time Complexity: O(T * N) where T is the number of time steps and N is the length of the queue string.
 *                  The outer loop runs T times, and the inner loop runs up to N times.
 * Space Complexity: O(N) to store the queue string and its copy.
 */

/*
Link: https://codeforces.com/problemset/problem/266/B
@uthor: chavodon
*/ // Preserving original author's comments as per rules.

#include <iostream> // Required for standard input/output operations.
// #include <string> // Implicitly included with iostream or bits/stdc++.h

/**
 * @brief Standard namespace for C++ functionalities.
 */
using namespace std;

/**
 * @brief Main function where the program execution begins.
 * @return int Returns 0 on successful execution.
 */
int main() {
  string s; // Original input string representing the queue.
  string a; // Modifiable copy of the string for simulation.
  int n;    // Length of the queue string.
  int t;    // Number of time steps for the simulation.
  int i, j; // Loop counters.

  /**
   * @brief Reads the queue length `n`, number of time steps `t`, and the initial queue string `s`.
   *
   * Pre-condition: `n`, `t` integers and string `s` are available on standard input.
   * Post-condition: Variables `n`, `t`, `s` are initialized with input values.
   */
  cin >> n >> t;
  cin >> s;
  a = s; // Functional Utility: Creates a modifiable copy of the initial queue string.

  /**
   * @brief Outer loop: Simulates the queue transformation over `t` time steps.
   *
   * Invariant: `i` counts the completed time steps. The loop continues for `t` iterations.
   * Pre-condition: `t` specifies the total simulation duration.
   * Post-condition: The queue `a` reflects its state after `t` time steps.
   */
  for (i = 0; i < t; i++) {
    /**
     * @brief Inner loop: Performs a single pass through the queue to apply swaps for the current time step.
     *
     * Invariant: `j` traverses the queue. Swaps are performed locally, and `j` is incremented to ensure
     *            a boy moves only one position per time step.
     * Pre-condition: `a` holds the current state of the queue.
     * Post-condition: `a` reflects the queue state after one time step's movements.
     */
    for (j = 1; j < n; j++) {
      /**
       * @brief Conditional Logic: Checks for a "BG" pattern to perform a swap.
       *
       * Functional Utility: Swaps a boy ('B') and a girl ('G') if the boy is immediately in front of the girl.
       * Inline: `j++` ensures that the newly swapped 'G' is not re-evaluated immediately, preventing
       *         a boy from "jumping" multiple positions in one time step.
       */
      if (a[j - 1] == 'B' && a[j] == 'G') {
        a[j - 1] = 'G';
        a[j] = 'B';
        j++; // Skips the current position as the 'B' has moved, and 'G' is now there.
      }
    }
  }
  /**
   * @brief Prints the final state of the queue after all time steps.
   *
   * Pre-condition: `a` contains the final arrangement of the queue.
   * Post-condition: The final queue string is displayed on standard output.
   */
  cout << a;
  return 0; // Ensures the program exits successfully.
}