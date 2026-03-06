/**
 * @file 339A.CPP
 * @brief Sorts a given mathematical sum of 1s, 2s, and 3s.
 *
 * This program takes a string representing a sum (e.g., "3+2+1+1") and
 * rearranges the numbers in non-decreasing order.
 *
 * @algorithm Counting Sort
 * @complexity
 * Time: O(N), where N is the length of the input string, because we do a single pass
 * to count the numbers and another pass to construct the output string.
 * Space: O(1) auxiliary space, as the counts array is of fixed size (3), and the
 * output string length is proportional to the input string length.
 */
#include <iostream>
#include <string>

/**
 * @brief Parses the input string and counts the occurrences of '1', '2', and '3'.
 * @param s The input string representing the sum.
 * @param counts An array of integers to store the frequency of numbers.
 *               counts[0] for '1's, counts[1] for '2's, counts[2] for '3's.
 * @note The function modifies the counts array in place.
 */
void parse(std::string &s, int counts[]) {
  // Pre-condition: The 'counts' array is initialized to zeros.
  // Block Logic: Iterate through the string. If a character is a digit,
  // increment the corresponding counter. Other characters (like '+') are ignored.
  for (int i = 0; i < s.size(); ++i) {
    if (s[i] == '1')
      ++counts[0];
    else if (s[i] == '2')
      ++counts[1];
    else if (s[i] == '3')
      ++counts[2];
  }
}

/**
 * @brief Constructs and prints the sorted sum string from the counts.
 * @param counts An array containing the frequency of '1's, '2's, and '3's.
 */
void printOutput(int counts[]) {
  std::string result = "";
  int total_numbers = counts[0] + counts[1] + counts[2];
  int numbers_added = 0;

  // Pre-condition: counts array contains the frequency of each number.
  // Block Logic: Append the '1's to the result string, separated by '+'.
  for (int i = 0; i < counts[0]; ++i) {
    result += '1';
    numbers_added++;
    if (numbers_added < total_numbers) {
      result += '+';
    }
  }

  // Block Logic: Append the '2's to the result string, separated by '+'.
  for (int i = 0; i < counts[1]; ++i) {
    result += '2';
    numbers_added++;
    if (numbers_added < total_numbers) {
      result += '+';
    }
  }

  // Block Logic: Append the '3's to the result string, separated by '+'.
  for (int i = 0; i < counts[2]; ++i) {
    result += '3';
    numbers_added++;
    if (numbers_added < total_numbers) {
      result += '+';
    }
  }

  std::cout << result << '
';
}

/**
 * @brief Main function to read the input, process it, and print the output.
 */
int main() {
  std::string s;
  std::cin >> s;
  int counts[3] = {0};
  // Step 1: Count the frequency of each number.
  parse(s, counts);
  // Step 2: Reconstruct and print the sorted sum.
  printOutput(counts);
}
