/**
 * @file 339A.CPP
 * @brief Solution to a competitive programming problem (Codeforces 339A - "Helpful Maths").
 *
 * The problem requires sorting a sum of numbers (1, 2, and 3) given as a string,
 * e.g., "1+3+2", into ascending order, e.g., "1+2+3".
 *
 * Algorithm: Counting Sort.
 * The solution counts the occurrences of each digit ('1', '2', '3') in the input string.
 * Then, it constructs the output string by appending '1's, followed by '2's, and then '3's,
 * each separated by a '+' sign. The final '+' is removed before printing.
 *
 * Time Complexity: O(L), where L is the length of the input string.
 * This includes one pass to count digits and a subsequent pass to build the result string.
 * Space Complexity: O(L) to store the resulting sorted sum string.
 */
// 339A
#include <cstring>
#include <iostream>
#include <string> // Required for std::string
using namespace std;

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads an arithmetic expression string, sorts its terms, and prints the sorted expression.
 */
int main() {
  string s, r; // 's' for input string, 'r' for result string.
  // Pre-condition: Read the input sum string 's'.
  cin >> s;
  // Initialize counters for digits '1', '2', and '3'.
  int count1 = 0, count2 = 0, count3 = 0;
  // Block Logic: Iterate through the input string to count occurrences of each digit.
  // Invariant: 'i' traverses the string 's'.
  for (int i = 0; i < s.length(); i++) {
    // If the character is a digit, increment the corresponding counter.
    if (s[i] == '1') {
      count1++;
    } else if (s[i] == '2') {
      count2++;
    } else if (s[i] == '3') {
      count3++;
    }
  }
  // Block Logic: Append '1's and '+' to the result string 'r' based on count1.
  for (int i = 0; i < count1; i++) {
    r.push_back('1');
    r.push_back('+');
  }
  // Block Logic: Append '2's and '+' to the result string 'r' based on count2.
  for (int i = 0; i < count2; i++) {
    r.push_back('2');
    r.push_back('+');
  }
  // Block Logic: Append '3's and '+' to the result string 'r' based on count3.
  for (int i = 0; i < count3; i++) {
    r.push_back('3');
    r.push_back('+');
  }
  // Block Logic: Print the resulting string, excluding the last '+' character.
  // Pre-condition: The string 'r' is not empty (it will always have at least one digit and one '+').
  // Invariant: 'i' iterates up to the second to last character of 'r'.
  for (int i = 0; i < r.length() - 1; i++)
    cout << r[i];
  cout << endl; // Add a newline at the end of the output.
  return 0;
}
