/**
 * @file 339A.CPP
 * @brief This program reads a string representing an arithmetic sum of numbers
 * separated by '+', sorts the numbers in non-decreasing order, and then prints
 * them back as a sum. It's a common problem in competitive programming, focusing
 * on string parsing, integer conversion, sorting, and string reconstruction.
 *
 * The overall design involves:
 * 1. Reading the input string.
 * 2. Parsing the string to extract individual numbers.
 * 3. Storing these numbers in a vector.
 * 4. Sorting the vector.
 * 5. Printing the sorted numbers, re-inserting '+' signs between them.
 *
 * Time Complexity: O(N log N) due to sorting, where N is the number of integers.
 * Space Complexity: O(N) for storing the integers in a vector.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function where the program execution begins.
 *
 * This function handles the entire process:
 * - Reads the input string.
 * - Parses integers from the string.
 * - Sorts the extracted integers.
 * - Prints the sorted integers as a sum.
 *
 * @return 0 if the program executes successfully.
 */
int main() {
  string s;
  int cnt = 0;
  vector<int> vec;
  getline(cin, s); // Read the entire line containing the sum (e.g., "1+5+2")
  stringstream ss(s); // Create a stringstream to parse the input string
  string ind;
  /**
   * @brief Loop to extract integers from the input string.
   *
   * This while loop reads substrings separated by the '+' delimiter.
   * Each substring is then converted to an integer and added to the 'vec' vector.
   *
   * Pre-condition: 's' contains a string of numbers separated by '+'.
   * Invariant: 'vec' contains all numbers parsed from 's' up to the current point.
   */
  while (getline(ss, ind, '+')) {
    int a = stoi(ind); // Convert the extracted substring to an integer
    vec.push_back(a);  // Add the integer to the vector
  }
  /**
   * @brief Sorts the vector of integers.
   *
   * This function call sorts the numbers in 'vec' in non-decreasing order.
   *
   * Time Complexity: O(N log N)
   * Pre-condition: 'vec' contains all parsed integers.
   * Post-condition: 'vec' contains the same integers, but sorted.
   */
  sort(vec.begin(), vec.end());
  cout << vec[0]; // Print the first element, as it won't have a leading '+'
  /**
   * @brief Loop to print the sorted integers as a sum.
   *
   * This for loop iterates through the sorted vector starting from the second
   * element, printing a '+' before each number to reconstruct the sum string.
   *
   * Pre-condition: 'vec' is sorted and its first element has been printed.
   * Invariant: The sum string has been reconstructed with sorted numbers up to 'i-1'.
   */
  for (int i = 1; i < vec.size(); i++) {
    cout << "+" << vec[i]; // Prepend '+' for subsequent numbers
  }
  cout << endl; // Print a newline at the end

  return 0;
}
