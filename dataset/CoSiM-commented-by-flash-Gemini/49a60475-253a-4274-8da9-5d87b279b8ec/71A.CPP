/**
 * @49a60475-253a-4274-8da9-5d87b279b8ec/71A.CPP
 * @brief This program abbreviates words that are longer than 10 characters,
 *        by printing the first letter, the count of intermediate characters,
 *        and the last letter. Shorter words are printed as is.
 * Algorithm: String length check and conditional printing based on length.
 * Time Complexity: O(N * L) where N is the number of words and L is the
 *                  average length of a word, due to string length calculation and character access.
 * Space Complexity: O(L) for storing each input word.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {

  int n; // Declare 'n' to store the number of words to process.
  // Functional Utility: Read the number of words.
  cin >> n;
  string a; // Declare a string 'a' to store each input word.
  // Block Logic: Iterate 'n' times, processing each word.
  for (int i = 0; i < n; i++) {
    // Functional Utility: Read the current word.
    cin >> a;
    // Pre-condition: Check if the word's length is greater than 10.
    if (a.length() > 10) {
      // Action: Print the abbreviated form: first character, length-2, last character.
      cout << a[0] << a.length() - 2 << a[a.length() - 1] << endl;
    } else {
      // Action: Otherwise, print the word as is.
      cout << a << endl;
    }
  }

  return 0; // Indicate successful program termination.
}
