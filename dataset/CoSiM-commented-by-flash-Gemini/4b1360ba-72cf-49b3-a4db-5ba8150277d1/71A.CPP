/**
 * @4b1360ba-72cf-49b3-a4db-5ba8150277d1/71A.CPP
 * @brief This program abbreviates words that are longer than 10 characters,
 *        by printing the first letter, the count of intermediate characters,
 *        and the last letter. Shorter words are printed as is.
 * Algorithm: String length check and conditional printing based on length,
 *            using `string::front()` and `string::back()` for character access.
 * Time Complexity: O(N * L) where N is the number of words and L is the
 *                  average length of a word, due to string length calculation and character access.
 * Space Complexity: O(L) for storing each input word.
 */
#include <iostream>
#include <string>
using namespace std;

int main() {
  int n; // Declare 'n' to store the number of words to process.
  string input; // Declare a string 'input' to store each input word.
  // Functional Utility: Read the number of words.
  cin >> n;
  // Block Logic: Iterate 'n' times, processing each word.
  while (n-- > 0) { // Loop 'n' times, decrementing 'n' in each iteration.
    // Functional Utility: Read the current word.
    cin >> input;

    // Pre-condition: Check if the word's length is greater than 10.
    if (input.length() > 10)
      // Action: Print the abbreviated form: first character, length-2, last character.
      cout << input.front() << input.length() - 2 << input.back() << endl;
    else
      // Action: Otherwise, print the word as is.
      cout << input << endl;
  }
  return 0; // Indicate successful program termination.
}