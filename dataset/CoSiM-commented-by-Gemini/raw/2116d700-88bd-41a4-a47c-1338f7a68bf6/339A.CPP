/**
 * @file 339A.CPP
 * @brief Solution for Codeforces Problem 339A, "Helpful Maths".
 * @details This program takes a string representing a sum of numbers (1, 2, and 3)
 * and rearranges it into a non-decreasing sequence. For example, "3+2+1"
 * becomes "1+2+3".
 *
 * @algorithm Frequency Counting and String Reconstruction
 * @description The program first counts the occurrences of each number (1, 2, 3)
 * using a map. It then reconstructs the string by first adding the smallest
 * available number, and then appending the rest of the numbers in increasing
 * order, separated by plus signs.
 * @complexity
 * - Time: O(N), where N is the length of the input string, as both parsing and
 *   reconstruction are linear operations.
 * - Space: O(N) for storing the output string. The map uses O(1) space as it
 *   contains at most three keys.
 */
#include <iostream>
#include <map>
#include <sstream>
#include <string>
using namespace std;
int main() {
  // m1 stores the frequency of the numbers 1, 2, and 3.
  map<int, int> m1;
  m1[1] = 0;
  m1[2] = 0;
  m1[3] = 0;
  string s;
  cin >> s;

  // Iterate through the input string to populate the frequency map.
  // The characters '+' are ignored.
  for (int i = 0; i < s.length(); i++)
    if (s[i] != '+')
      m1[s[i] - '0'] += 1;
  
  string g = "";

  /**
   * @brief This block constructs the first term of the new sum.
   * It finds the smallest number with a non-zero count and appends it to the
   * result string `g`. This ensures the first number is not preceded by a '+'.
   */
  if (m1[1] > 0) {
    g += "1";
    m1[1] -= 1;
  } else if (m1[2] > 0) {
    g += "2";
    m1[2] -= 1;
  } else {
    g += "3";
    m1[3] -= 1;
  }

  /**
   * @brief Appends the remaining numbers to the result string in non-decreasing order.
   * This loop iterates through the numbers 1, 2, and 3. For each number, it appends
   * "+<number>" to the string as many times as it was counted.
   */
  for (int i = 1; i <= 3; i++) {
    char y = i + '0';
    string d(1, y);
    string x = "+" + (d);
    while (m1[i] > 0) {
      g = g + x;
      m1[i] -= 1;
    }
  }
  cout << g << endl;
}