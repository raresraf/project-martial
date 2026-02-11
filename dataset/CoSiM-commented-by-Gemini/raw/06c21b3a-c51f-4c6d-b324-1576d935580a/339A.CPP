/**
 * @file 339A.CPP
 * @brief Rearranges a sum of numbers to be in non-decreasing order.
 *
 * This program takes a string representing a sum of single-digit numbers (e.g., "3+1+2"),
 * parses the numbers, sorts them, and prints them back in non-decreasing order (e.g., "1+2+3").
 * This is a solution to the "Helpful Maths" problem.
 *
 * @see https://codeforces.com/problemset/problem/339/A
 *
 * Algorithm:
 * 1. Read the input string containing the sum.
 * 2. Parse the string, extracting the digits (which are at even indices).
 * 3. Convert the character digits to integers and store them in an array.
 * 4. Sort the array of numbers in non-decreasing order.
 * 5. Print the sorted numbers, separated by '+' signs.
 *
 * Time Complexity: O(N log N), where N is the length of the input string. The sorting step dominates the complexity.
 * Space Complexity: O(N) to store the extracted numbers in an array.
 */
#include <algorithm>
#include <iostream>
using namespace std;

// a: the input string representing the sum.
string a;
// mas: an array to store the parsed numbers.
// b: a counter for the number of elements in mas.
int mas[105], b;

int main() {
  // Read the sum as a string.
  cin >> a;

  /**
   * @brief Parses the string to extract the numbers.
   *
   * The numbers in the string are located at even indices (0, 2, 4, ...).
   * The character is converted to an integer by subtracting the ASCII value of '0' (48).
   */
  for (int i = 0; i < a.size(); i += 2) {
    mas[b] = a[i] - 48;
    b++;
  }

  // Sort the array of numbers. The number of elements is (a.size() / 2 + 1).
  sort(mas, mas + a.size() / 2 + 1);

  // Print the first number.
  cout << mas[0];

  /**
   * @brief Prints the rest of the numbers, separated by '+'.
   *
   * Pre-condition: The 'mas' array is sorted.
   * Invariant: The first 'i' numbers have been printed in sorted order.
   */
  for (int i = 1; i < a.size() / 2 + 1; i++) {
    cout << "+" << mas[i];
  }
}
