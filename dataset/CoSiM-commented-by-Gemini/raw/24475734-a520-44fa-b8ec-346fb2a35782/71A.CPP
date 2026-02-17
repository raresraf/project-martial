/**
 * @file 71A.CPP
 * @brief Solution for Codeforces Problem 71A, "Way Too Long Words".
 * @details This program processes a list of words. If a word is longer than 10 characters,
 * it is replaced by a special abbreviation. The abbreviation consists of the first
 * letter, the number of letters between the first and last, and the last letter.
 * Otherwise, the word is printed as is.
 *
 * @algorithm String manipulation and conditional logic.
 * @time_complexity O(N * L_avg), where N is the number of words and L_avg is their
 * average length. The complexity is dominated by the input/output operations.
 * @space_complexity O(L_max), where L_max is the maximum length of a single word.
 */
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
using std::cin;
using std::cout;
using std::ostringstream;
using std::string;

int main() {
  int n;
  string word;
  cin >> n; // Read the number of words to process.

  /**
   * Block Logic: Loop 'n' times to process each word individually.
   */
  for (int i = 0; i < n; ++i) {
    cin >> word;
    
    // Pre-condition: Check if the word's length exceeds the threshold for abbreviation.
    if (word.size() > 10) {
      // Use a stringstream to convert the integer length into a string.
      ostringstream ss;
      ss << (word.size() - 2);
      
      // Construct the abbreviated word.
      word = word.front() + ss.str() + word.back();
    }
    // Output the original or abbreviated word.
    cout << word << "\n";
  }
  return 0;
}