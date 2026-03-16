/**
 * @file 266B.CPP
 * @brief Solution to the "Queue at the School" problem (Codeforces 266B).
 *
 * @details
 * This program simulates the changes in a school queue composed of boys ('B')
 * and girls ('G'). In each time step, any boy standing immediately before a girl
 * swaps places with her. The program calculates the final state of the queue
 * after a given number of time steps.
 *
 * The core logic is encapsulated within the `findOrder` function, which performs
 * the step-by-step simulation. This separation of logic from I/O is good practice.
 * The simulation itself is identical to other correct solutions for this problem:
 * a nested loop structure where the outer loop tracks time and the inner loop
 * performs the swaps for a single time step.
 *
 * Algorithm: Iterative single-pass swap simulation.
 * Time Complexity: O(t * n), where `t` is the number of time steps and `n` is the
 *                  queue length.
 * Space Complexity: O(n) to store the queue string.
 */
#include <iostream>
#include <string>
using namespace std;

/**
 * @brief Simulates the queue changes over a given number of time steps.
 * @param a The initial state of the queue string. Passed by value.
 * @param n The number of children in the queue.
 * @param j The number of time steps to simulate.
 * @return The final state of the queue string after `j` time steps.
 */
string findOrder(string a, int n, int j) {
  // Outer loop simulates the passage of time.
  while (j--) {
    int i = 0;
    // Inner loop performs a single pass to find and execute all swaps for the current time step.
    while (i < n - 1) {
      if (a[i] == 'B' && a[i + 1] == 'G') {
        // A boy is in front of a girl, so they swap.
        a[i] = 'G';
        a[i + 1] = 'B';
        // Inline: Advance index by 2. This is crucial to ensure that after a swap,
        // the now-swapped boy at `i+1` isn't immediately swapped again in the same
        // time step, correctly modeling the problem's constraints.
        i += 2;
      } else {
        // If no swap, move to the next position.
        i++;
      }
    }
  }
  return a;
}

/**
 * @brief Main function to handle input/output.
 */
int main(int argc, char const *argv[]) {
  int n, j;
  cin >> n >> j;
  string str;
  cin >> str;
  cout << findOrder(str, n, j);
  return 0;
}