/**
 * @file 339A.CPP
 * @brief Solution for the "Helpful Maths" problem (Codeforces 339A).
 * @author competitive-programmer
 *
 * @details This program takes a string representing a sum of numbers (1, 2, and 3),
 * and reorders it so that the numbers are in non-decreasing order. For example,
 * an input of "3+2+1" should produce the output "1+2+3".
 *
 * Algorithm:
 * 1. Read the input string which represents the sum.
 * 2. The `sorting` function first attempts to remove the '+' characters.
 *    It then uses a form of counting sort by segregating the numbers '1', '2',
 *    and '3' into separate strings.
 * 3. These separate strings are concatenated to form a single string of sorted numbers.
 * 4. The `print` function takes this sorted string of numbers and reconstructs
 *    the sum format with '+' signs for the final output.
 *
 * Time Complexity: O(N), where N is the length of the input string.
 * Space Complexity: O(N), to store the intermediate strings for '1's, '2's, and '3's.
 */
#include <iostream>
#include <string>
#include <algorithm> // For a more robust sorting solution

using namespace std;

string sorting(string input);
void print(string sorted);

int main() {
  string input;
  string sorted = "";
  cin >> input;

  // Calls the sorting function to process the string and then prints the result.
  sorted = sorting(input);
  print(sorted);
  return 0;
}

/**
 * @brief Sorts the numbers in the input string.
 * @param input The string representing the sum, e.g., "1+3+2".
 * @return A string with the numbers sorted, e.g., "123".
 */
string sorting(string input) {
  string ones, twos, threes;

  // @note Bug Warning: Modifying a string while iterating over it with index-based
  // access can lead to incorrect behavior (e.g., skipping characters).
  // A safer approach would be to extract the numbers first and then sort.
  for (int i = 1; i < input.length(); i += 2) {
    if (input.at(i) == '+') {
      input.erase(i, 1);
    }
  }

  /**
   * Block Logic: A form of counting sort. It iterates through the string of
   * numbers (with '+' removed) and appends each number to one of three
   * separate strings based on its value.
   */
  for (int i = 0; i < input.length(); i++) {
    if (input.at(i) == '1') {
      ones = ones + input.at(i);
    } else if (input.at(i) == '2') {
      twos = twos + input.at(i);
    } else if (input.at(i) == '3') {
      threes = threes + input.at(i);
    }
  }
  
  // The final sorted string is a concatenation of the '1's, '2's, and '3's.
  return ones + twos + threes;
}

/**
 * @brief Prints the sorted numbers in the format of a sum.
 * @param sorted A string containing the sorted digits, e.g., "123".
 */
void print(string sorted) {
  for (int i = 0; i < sorted.length(); i++) {
    cout << sorted.at(i);
    // Adds a '+' between the numbers, but not after the last one.
    if (i != sorted.length() - 1) {
      cout << "+";
    }
  }
}
