// solutions.cpp : This file contains the 'main' function. Program execution
// begins and ends there.
//
// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started:
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see buil4d output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add
//   Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project
//   and select the .sln file

/**
 * @file 266B.CPP
 * @brief Solution to a competitive programming problem: "Queue at the School".
 *
 * This program simulates the reordering of a queue where 'B' (boys) and 'G' (girls)
 * swap positions. Specifically, if a girl ('G') is ahead of a boy ('B'), they swap,
 * effectively moving the 'G' backward (or the 'B' forward). This process is
 * repeated for a given number of time steps.
 *
 * Algorithm: Discrete-time simulation with a single-pass swap rule, focusing on moving 'G's backward.
 * Time Complexity: O(T * N), where T is the number of time steps and N is the length of the queue (string).
 *                  Each time step involves a single pass through the queue.
 * Space Complexity: O(N) to store the queue string.
 */
#include <algorithm> // Required for std::swap.
#include <bitset>    // Potentially for bit manipulation, though not used in main.
#include <cmath>     // Potentially for mathematical functions, used in isprime.
#include <iostream>  // Required for input/output operations (cin, cout).
#include <queue>     // Potentially for queue data structure, though not used in main.
#include <stack>     // Potentially for stack data structure, though not used in main.
#include <utility>   // Required for std::pair (used by 'coord').
#include <vector>    // Required for std::vector.

// Macros for common operations and type definitions, optimizing competitive programming code.
#define mp make_pair ///< @brief Creates a std::pair.
#define pb push_back ///< @brief Appends an element to a std::vector.
#define f first      ///< @brief Accesses the first element of a std::pair.
#define s second     ///< @brief Accesses the second element of a std::pair.

/**
 * @brief Standard namespace usage.
 * In competitive programming, 'using namespace std;' is common practice to avoid
 * repeatedly prefixing standard library elements with 'std::'.
 */
using namespace std;

// Type definitions for various data types.
typedef long long ll;         ///< @brief Alias for long long integer.
typedef unsigned long long ull; ///< @brief Alias for unsigned long long integer.
typedef pair<int, int> coord; ///< @brief Alias for a pair of integers, possibly representing coordinates.
typedef vector<int> veci;     ///< @brief Alias for a vector of integers.
typedef vector<coord> coords; ///< @brief Alias for a vector of coordinate pairs.
const int mod = 1000000007;   ///< @brief Constant for modulo operation.
const int mod2 = 998244353;   ///< @brief Another constant for modulo operation.

/**
 * @brief Checks if a given unsigned long long number is prime.
 *
 * This function implements a trial division algorithm to determine primality.
 * It handles small primes (2, 3) specifically and then iterates through
 * potential divisors up to the square root of the number.
 *
 * Algorithm: Trial division.
 * Time Complexity: O(sqrt(n)).
 * @param n The number to check for primality.
 * @return True if 'n' is prime, false otherwise.
 */
bool isprime(ull n) {
  if (n % 2 == 0) { // Check for divisibility by 2.
    return n == 2;  // 2 is prime, other even numbers are not.
  }
  if (n % 3 == 0) { // Check for divisibility by 3.
    return n == 3;  // 3 is prime, other multiples of 3 are not.
  }
  ull k = sqrt(n); // Calculate the square root of n for optimization.
  for (int i = 5; i < k; i += 2) { // Iterate through odd numbers starting from 5.
    if (n % k == 0) { // If divisible, it's not prime.
      return false;
    }
  }
  return n != 1; // 1 is not prime.
}

/**
 * @brief Main function: entry point of the program.
 * Orchestrates the simulation of the queue reordering process based on the
 * given rules and prints the final state of the queue.
 */
int main() {
  ios_base::sync_with_stdio(false); // Disables synchronization with C standard streams for fast I/O.
  string line;                      // Declares 'line' to represent the queue string.
  int num, time;                    // 'num' for queue length, 'time' for number of time steps.
  cin >> num >> time >> line;       // Reads queue length, time steps, and initial queue state.

  /**
   * @brief Simulates the queue reordering over 'time' steps.
   * Block Logic: Each iteration represents one second in the simulation, applying the swap rule.
   * Invariant: 'time' decreases with each iteration, ensuring the simulation runs for the specified duration.
   */
  while (time--) {
    /**
     * @brief Performs a single-pass sweep to apply the 'BG' to 'GB' swap rule.
     * Block Logic: Iterates through the queue, identifying 'G's that can swap with a preceding 'B'.
     * Pre-condition: 'line' represents the current state of the queue for the current time step.
     */
    for (int i = 0; i < num; ++i) {
      // Condition for swapping: current character is 'G' and it's not the first element (i > 0).
      // This implies the preceding character `line[i-1]` must be 'B' for a meaningful swap,
      // which results in 'GB' becoming 'BG'.
      if (line[i] == 'G' && i) {
        swap(line[i], line[i - 1]); // Swaps the current 'G' with the character before it.
        ++i;                       // Inline: Increment 'i' again to skip the newly moved 'G'.
                                   // This is crucial to ensure that a 'B' (which was `line[i]`)
                                   // only moves one position forward per second and does not
                                   // "jump" multiple 'G's in a single time step.
      }
    }
  }
  cout << line << '\n'; // Prints the final state of the queue after all time steps.
}