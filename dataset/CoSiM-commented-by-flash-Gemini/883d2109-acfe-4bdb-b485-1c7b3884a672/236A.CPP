/**
 * @file 236A.CPP
 * @brief Solves the "Boy or Girl" problem from Codeforces (Problem 236A).
 * This program determines if a given username (string) corresponds to a "boy" or "girl"
 * based on the parity of the number of distinct characters in the username.
 *
 * Algorithm: Count distinct characters using a hash map (std::map). The size of
 * the map after processing the string gives the count of distinct characters.
 * Parity check on this count determines the output.
 * Time Complexity: O(L log K), where L is the length of the string and K is the
 *                  number of distinct characters (or alphabet size). Map insertion
 *                  takes logarithmic time.
 * Space Complexity: O(K) for storing distinct characters in the map.
 */

#include <iostream> // @brief Required for standard input/output streams (cin, cout).
#include <map>      // @brief Required for std::map to count character frequencies.
#include <string>   // @brief Required for std::string.

using namespace std;

int main() {
  /**
   * @brief Main function to solve the Boy or Girl problem.
   * Reads a username string, counts its distinct characters using a map, and determines
   * if the count is odd ("IGNORE HIM!") or even ("CHAT WITH HER!").
   */

  string cad; // @brief The input username string.
  map<char, int> caracteres; // @brief Map to store character frequencies (key: character, value: count).
  cin >> cad; // @brief Read the input username string.
  
  // Block Logic: Count distinct characters by populating the map.
  // Each unique character in the string becomes a key in the map,
  // and its associated value is incremented, effectively counting occurrences.
  // The size of the map (`caracteres.size()`) then represents the number of distinct characters.
  for (int i = 0; i < cad.size(); i++) {
    caracteres[cad[i]]++; // Inline: Increment count for the current character.
  }

  // Block Logic: Determine the output based on the parity of the distinct character count.
  // If `caracteres.size()` (number of distinct characters) is odd, it's a "boy".
  if (caracteres.size() & 1) { // Inline: Bitwise AND with 1 checks for oddness.
    cout << "IGNORE HIM!" << endl;
  } 
  // Otherwise, it's a "girl".
  else {
    cout << "CHAT WITH HER!" << endl;
  }

  return 0; // @brief Indicates successful program termination.
}