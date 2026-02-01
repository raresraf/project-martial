/**
 * @file 339A.CPP
 * @brief This program sorts a mathematical expression consisting of numbers separated by '+' signs.
 * The input is a string representing an addition problem, and the output is the same problem
 * with the numbers sorted in non-decreasing order.
 * Algorithm: String parsing, integer conversion, sorting, and string reconstruction.
 * Time Complexity: O(N log N) where N is the count of numbers in the input string due to sorting.
 * Space Complexity: O(N) to store the parsed numbers.
 */

#include <algorithm> // Required for std::sort.
#include <iostream>  // Required for input/output operations (cin, cout).
#include <string>    // Required for string manipulation.
#include <vector>    // Required for dynamic array (vector) to store numbers.

using namespace std;

int main() {
  string delimiter = "+"; // Defines the delimiter used to separate numbers in the input string.
  string input;           // Stores the input mathematical expression.
  size_t pos = 0;         // Stores the position of the delimiter found in the input string.
  string token;           // Temporarily stores a parsed number (substring).
  vector<int> numbers;    // Dynamically sized array to store integer representations of the parsed numbers.

  // Functional Utility: Reads the entire input line into the 'input' string.
  cin >> input;

  /*
   * Block Logic: Parses the input string to extract individual numbers.
   * Invariant: At the start of each iteration, 'input' contains the remaining unsplit part
   * of the original string, and 'numbers' contains all previously extracted integers.
   * The loop continues as long as the delimiter is found within the 'input' string.
   */
  while ((pos = input.find(delimiter)) != string::npos) {
    // Extracts the substring before the found delimiter, representing a number.
    token = input.substr(0, pos);
    // Converts the extracted number string to an integer and adds it to the 'numbers' vector.
    numbers.push_back(stoi(token));
    // Removes the processed number and its delimiter from the 'input' string, preparing for the next iteration.
    input.erase(0, pos + delimiter.length());
  }
  // Block Logic: Handles the last number in the string after all delimiters have been processed.
  // Pre-condition: 'input' contains only the last number (or the entire string if no delimiters were present).
  numbers.push_back(stoi(input)); // Converts and adds the last remaining number to the vector.

  // Functional Utility: Sorts the vector of extracted integers in non-decreasing order.
  // Algorithm: Typically Introsort (a hybrid algorithm) in libstdc++.
  sort(numbers.begin(), numbers.end());

  string output; // String to build the sorted mathematical expression.
  // Block Logic: Reconstructs the mathematical expression with sorted numbers.
  // Invariant: 'output' accumulates sorted numbers, each followed by a '+', except for the last one.
  for (int i : numbers) {
    output.append(to_string(i) + "+"); // Appends the number and a '+' to the output string.
  }
  // Post-condition: Removes the trailing '+' character that is appended after the last number.
  output.erase(output.length() - 1);

  // Functional Utility: Prints the final sorted mathematical expression to standard output.
  std::cout << output << endl;
}
