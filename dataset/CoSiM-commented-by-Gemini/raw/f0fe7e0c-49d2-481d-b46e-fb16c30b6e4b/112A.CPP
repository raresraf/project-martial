/**
 * @file 112A.CPP
 * @brief Performs a case-insensitive lexicographical comparison of two strings.
 *
 * This program reads two strings, converts them to uppercase, and then compares
 * them lexicographically. It prints -1 if the first string is less than the
 * second, 1 if it's greater, and 0 if they are equal.
 *
 * @b Algorithm: The program iterates through the strings character by character.
 *    In each iteration, it converts the characters from both strings to uppercase
 *    by checking if they are in the lowercase range 'a'-'z'. It then compares
 *    these characters. If a difference is found, the result is stored and the
 *    loop terminates.
 *
 * @b Time_Complexity: O(L), where L is the length of the strings, as it may
 *    iterate through the entire length of the strings.
 * @b Space_Complexity: O(L) to store the two input strings.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  string first;
  string second;
  int diff = 0;
  cin >> first;
  cin >> second;

  /**
   * @brief This loop iterates through the strings, comparing them character by character
   * after converting them to uppercase.
   */
  for (int i = 0; i < first.length(); i++) {
    // Convert first string's character to uppercase if it's lowercase.
    char temp = first.at(i);
    if (temp >= 'a' && temp <= 'z') {
      first[i] = temp - 32; // Subtracting 32 converts from lowercase to uppercase in ASCII.
    }
    
    // Convert second string's character to uppercase if it's lowercase.
    temp = second.at(i);
    if (temp >= 'a' && temp <= 'z') {
      second[i] = temp - 32;
    }
    
    // Compare the uppercase characters and break if a difference is found.
    if (first[i] > second[i]) {
      diff = 1;
      break;
    } else if (first[i] < second[i]) {
      diff = -1;
      break;
    }
  }
  cout << diff;
  return 0;
}
