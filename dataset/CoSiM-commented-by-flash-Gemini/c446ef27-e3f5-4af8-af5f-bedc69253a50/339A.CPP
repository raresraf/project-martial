/**
 * @c446ef27-e3f5-4af8-af5f-bedc69253a50/339A.CPP
 * @brief Solution for a competitive programming problem involving sorting digits from a sum.
 *
 * This program reads a string representing a sum (e.g., "1+5+2"), extracts the individual digits,
 * sorts them in non-decreasing order, and then prints the sorted digits re-interspersed with '+' signs.
 *
 * Algorithm:
 * 1. Read the input string.
 * 2. Iterate through the string, extracting characters at even indices (which are the digits).
 * 3. Store these digits in a vector.
 * 4. Sort the vector of digits.
 * 5. Print the sorted digits, adding a '+' sign between them for all but the last digit.
 *
 * Time Complexity: O(N log N) due to sorting, where N is the number of digits in the input string.
 * Space Complexity: O(N) to store the extracted digits in a vector.
 */

#include <algorithm> // Functional Utility: Provides the std::sort algorithm for sorting elements.
#include <iostream>  // Functional Utility: Provides input/output functionalities (std::cin, std::cout).
#include <vector>    // Functional Utility: Provides the std::vector container for dynamic arrays.
using namespace std;

int main() {
  string x;
  // Functional Utility: Reads the entire input string containing digits and '+' signs.
  cin >> x;
  int siz = x.size();
  vector<char> vec;
  // Block Logic: Iterates through the input string to extract only the digits.
  // Pre-condition: The input string 'x' contains digits at even indices and '+' at odd indices.
  // Invariant: 'j' increments by 2 to skip '+' characters, ensuring only digits are processed.
  for (int j = 0; j < siz;) {
    // Functional Utility: Adds the current character (a digit) to the vector.
    vec.push_back(x[j]);
    j += 2; // Inline: Skips the '+' character and moves to the next digit.
  }
  // Functional Utility: Sorts the collected digits in non-decreasing order.
  // Algorithm: Typically an IntroSort (hybrid of QuickSort, HeapSort, InsertionSort).
  sort(vec.begin(), vec.end());
  siz = vec.size();
  // Block Logic: Iterates through the sorted digits to print them with '+' signs.
  // Pre-condition: 'vec' contains sorted digits.
  // Invariant: Digits are printed in sorted order, separated by '+'.
  for (int j = 0; j < siz; j++) {
    cout << vec[j];
    // Conditional Logic: Prints a '+' separator after each digit except the last one.
    if (j + 1 < siz) {
      cout << "+";
    }
  }
  return 0; // Functional Utility: Indicates successful program execution.
}
