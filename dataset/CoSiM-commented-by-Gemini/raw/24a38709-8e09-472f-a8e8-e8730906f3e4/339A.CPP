/*
339A
Helpful maths
Xenia the beginner mathematician is a third year student at elementary school.
She is now learning the addition operation.

The teacher has written down the sum of multiple numbers. Pupils should
calculate the sum. To make the calculation easier, the sum only contains numbers
1, 2 and 3. Still, that isn't enough for Xenia. She is only beginning to count,
so she can calculate a sum only if the summands follow in non-decreasing order.
For example, she can't calculate sum 1+3+2+1 but she can calculate sums 1+1+2
and 3+3.

You've got the sum that was written on the board. Rearrange the summans and
print the sum in such a way that Xenia can calculate the sum.

Input
The first line contains a non-empty string s   the sum Xenia needs to count.
String s contains no spaces. It only contains digits and characters "+".
Besides, string s is a correct sum of numbers 1, 2 and 3. String s is at most
100 characters long.

Output
Print the new sum that Xenia can count.
*/
/* Accepted 60 ms 2000 Kb */

/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem by parsing, sorting, and rebuilding the sum string.
 * @details This implementation tokenizes the input string by the '+' delimiter,
 * converts the resulting number strings to integers, sorts the integers, and then
 * prints them back in the required sorted sum format.
 *
 * @algorithm String tokenization, type conversion, sorting, and string reconstruction.
 * @time_complexity O(L + N log N), where L is the string length and N is the number of summands (for sorting).
 * @space_complexity O(L) to store the intermediate string and integer vectors.
 */
#include <algorithm>
#include <iostream>
#include <vector>

// Note: This custom comparison function is defined but not used in the final code.
bool CfCompare(int i, int j) { return (i > j); }

/**
 * @brief Splits a string by a given delimiter.
 * @param s The string to be split.
 * @param c The delimiter character.
 * @return A vector of strings containing the tokens.
 */
const std::vector<std::string> explode(const std::string &s, const char &c) {
  std::string buff{ "" };
  std::vector<std::string> v;

  for (auto n : s) {
    if (n != c)
      buff += n;
    else if (n == c && buff != "") {
      v.push_back(buff);
      buff = "";
    }
  }
  if (buff != "")
    v.push_back(buff);

  return v;
}

int main() {
  std::string sum;
  std::cin >> sum;

  // Block Logic: Tokenize the input string into a vector of number strings.
  std::vector<std::string> vStr{explode(sum, '+')};
  
  // Block Logic: Convert the vector of number strings into a vector of integers.
  std::vector<int> v;
  std::transform(vStr.begin(), vStr.end(), std::back_inserter(v),
                 [](const std::string &str) { return std::stoi(str); });

  // Block Logic: Sort the numbers in non-decreasing order.
  std::sort(v.begin(), v.end());

  /**
   * Block Logic: Reconstruct and print the sorted sum string.
   * Invariant: At each step, an integer is printed. A '+' is appended for all
   * but the last integer in the vector.
   */
  for (std::vector<int>::iterator it = v.begin(); it != v.end(); ++it) {
    if (it != (v.end() - 1))
      std::cout << *it << "+";
    else
      std::cout << *it << '\n';
  }
  
  // Clean up memory.
  v.clear();
  vStr.clear();
  return 0;
}