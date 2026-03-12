/**
 * @5e6ded9e-9050-4b37-8593-5b6e8f18c0e6/116A.CPP
 * @brief This program attempts to solve problem 116A, "Tram".
 * It aims to determine the maximum number of passengers present in a tram
 * at any single moment during its journey through 'n' stops.
 * At each stop, passengers exit (`a`) and enter (`b`).
 *
 * NOTE: The implementation contains logical inconsistencies in calculating
 * the current number of passengers and the maximum observed, which may
 * lead to an incorrect solution for the problem.
 *
 * Algorithm: Reads all inputs into arrays, performs a partial initial calculation,
 * then iterates to update current and maximum passenger counts, followed by
 * an additional, potentially incorrect, pass over individual `a` and `b` values.
 * Time Complexity: O(N) where N is the number of tram stops.
 * Space Complexity: O(N) for storing input arrays.
 */

#include <iostream> // For standard input/output operations (std::cin, std::cout).

int main() {
  int n; // Declares 'n', representing the number of tram stops.
  std::cin >> n; // Reads 'n' from standard input.
  
  // Declares two arrays of size 'n' to store passengers exiting ('a') and entering ('b') at each stop.
  int a[n], b[n], max = 0, j; 
  // j is commented as "how many passengers there is after first trip", but its initialization seems off.
  
  /**
   * Block Logic: Reads all 'n' pairs of passenger exit/entry values into arrays.
   * Invariant: Arrays 'a' and 'b' are populated with the respective passenger counts.
   */
  for (int i = 0; i < n; i++) { 
    std::cin >> a[i] >> b[i]; // Reads passengers exiting and entering for the current stop.
  }

  // Block Logic: Initial calculation of 'j' (current passengers).
  // This initialization appears to be logically flawed for calculating current passengers after the first stop.
  // It effectively skips passengers exiting at the first stop (a[0]) and assumes the initial state is 0.
  // The expression 'b[0] - a[1] + b[1]' is inconsistent with sequential updates.
  j = b[0] - a[1] + b[1]; 

  /**
   * Block Logic: Iterates from the third stop onwards to update current passengers ('j')
   * and track the maximum observed ('max').
   * Invariant: 'max' stores the highest 'j' value encountered in this loop.
   */
  for (int i = 2; i < n; i++) { // Loop starts from the third stop (index 2).
    if (j > max) { // Inline: If current 'j' is greater than 'max', update 'max'.
      max = j;
    }
    // Inline: Updates 'j' by subtracting passengers exiting at current stop and adding those entering.
    // This calculation for 'j' is consistent for sequential updates, but the starting 'j' value is problematic.
    j = j - a[i] + b[i]; 
  }

  /**
   * Block Logic: This loop is potentially incorrect for determining the maximum *current* passengers.
   * It checks individual exit/entry numbers against the `max` calculated so far,
   * rather than the actual passenger count at any given stop.
   * Invariant: `max` might be updated by individual `a[i]` or `b[i]`, which is not the problem's goal.
   */
  for (int i = 0; i < n; i++) {
    if (a[i] > max) { // Inline: If passengers exiting at a stop is greater than max.
      max = a[i];
    }

    if (b[i] > max) { // Inline: If passengers entering at a stop is greater than max.
      max = b[i];
    }
  }

  std::cout << max; // Prints the final calculated maximum value.

  return 0; // Indicates successful program termination.
}