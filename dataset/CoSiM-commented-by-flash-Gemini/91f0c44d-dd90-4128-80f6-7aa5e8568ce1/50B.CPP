/**
 * @file 50B.CPP
 * @brief Calculates the sum of squares of character frequencies in an input string.
 *
 * This program reads a line of text from standard input, counts the occurrences
 * of each character (ASCII values 0-255), and then computes the sum of the
 * squares of these counts. This problem often appears in contexts where a metric
 * of character distribution or "diversity" is needed.
 *
 * Algorithm: Character frequency counting using an array indexed by ASCII values.
 *            Then, a linear scan to sum the squares of these frequencies.
 * Time Complexity: O(L + N), where L is the length of the input string and N is
 *                  the size of the character set (256 for ASCII).
 * Space Complexity: O(N) for the frequency array.
 */
#include <cstdio>   // For printf.
#include <iostream> // For std::cin, std::cout, getline.
#include <vector>   // For std::vector.

int main() {
  /**
   * @brief Main function: Reads a string, computes character frequencies, and sums the squares of frequencies.
   * @return 0 if the program executes successfully.
   */

  const int N = 256; // Defines the size of the character set (e.g., ASCII values).
  std::vector<long long> counts(N, 0); // Functional Utility: Vector to store character counts, initialized to zero.

  std::string input; // Declare a string to hold the input line.
  getline(std::cin, input); // Read an entire line of text from standard input.
  /**
   * @brief Loop to count the frequency of each character in the input string.
   * @pre 'input' contains the string, 'counts' vector is initialized to zeros.
   * @post 'counts[c]' stores the number of occurrences of character 'c'.
   * @invariant 'k' iterates through each character of the input string.
   */
  for (long k = 0; k < input.size(); k++) {
    ++counts[input[k]]; // Increment the count for the current character.
  }

  long long total(0); // Initialize a variable to store the sum of squares of counts.
  /**
   * @brief Loop to calculate the sum of squares of character frequencies.
   * @pre 'counts' vector contains the frequencies of all characters.
   * @post 'total' holds the sum of squares of all character frequencies.
   * @invariant 'k' iterates through all possible character ASCII values.
   */
  for (long k = 0; k < N; k++) {
    total += counts[k] * counts[k]; // Add the square of each character's count to the total.
  }

  printf("%lld\n", total); // Output the final sum of squares.

  return 0; // Return 0 for successful execution.
}
