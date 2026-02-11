/**
 * @file 339A.CPP
 * @brief Rearranges a sum of numbers to be in non-decreasing order.
 * @details This program solves the "Helpful Maths" problem from Codeforces. It takes a
 * string representing a sum (e.g., "3+1+2"), parses the numbers, sorts them,
 * and prints them back as a sum in non-decreasing order (e.g., "1+2+3").
 * This implementation uses standard C++ string manipulation and a vector.
 *
 * @see https://codeforces.com/problemset/problem/339/A
 *
 * Algorithm:
 * 1. Read the entire sum into a single string.
 * 2. Tokenize the string using '+' as a delimiter. Each token is a number.
 * 3. Convert each numeric token to an integer and store it in a std::vector.
 * 4. Sort the vector of integers in non-decreasing order.
 * 5. Build a new output string by iterating through the sorted vector, appending
 *    each number followed by a '+'.
 * 6. Remove the final trailing '+' from the output string.
 * 7. Print the result.
 *
 * Time Complexity: O(N log N), where N is the length of the input string. The complexity
 * is dominated by the sorting step on the parsed numbers.
 * Space Complexity: O(N), to store the parsed numbers in a vector and to build the
 * output string.
 */
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main() {
  string delimiter = "+";
  string input;
  size_t pos = 0;
  string token;
  vector<int> numbers;

  cin >> input;

  /**
   * @brief This loop finds instances of the delimiter '+'.
   *
   * It separates the numeric token, converts it to an integer,
   * pushes it into the `numbers` vector, and then erases the
   * token and the delimiter from the input string to process the remainder.
   */
  while ((pos = input.find(delimiter)) != string::npos) {
    token = input.substr(0, pos);
    numbers.push_back(stoi(token));
    input.erase(0, pos + delimiter.length());
  }
  // The final number is what's left of the input string after all delimiters are processed.
  numbers.push_back(stoi(input));
  
  // Sort the vector of numbers in non-decreasing order.
  sort(numbers.begin(), numbers.end());

  string output;
  // Reconstruct the output string from the sorted numbers.
  for (int i : numbers) {
    output.append(to_string(i) + "+");
  }
  
  // Remove the trailing "+" character at the end of the string.
  output.erase(output.length() - 1);

  std::cout << output << endl;
}
