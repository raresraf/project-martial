/**
 * @file 236A.CPP
 * @brief Solves the "Boy or Girl" problem using a std::map.
 *
 * @details
 * The program determines if a user is male or female based on the number of
 * distinct characters in their username. An even number of distinct characters
 * means the user is female ("CHAT WITH HER!"), while an odd number means male
 * ("IGNORE HIM!"). This implementation uses a `std::map` to efficiently count
 * the distinct characters.
 *
 * Algorithm:
 * 1. Read the input username string.
 * 2. Create a `std::map` to store the frequency of each character.
 * 3. Iterate through the string, and for each character, increment its count
 *    in the map. The map automatically handles unique characters.
 * 4. After the loop, the size of the map (`caracteres.size()`) gives the exact
 *    number of distinct characters.
 * 5. Check if the map's size is odd or even using a bitwise AND (`& 1`).
 * 6. Print "IGNORE HIM!" for an odd size and "CHAT WITH HER!" for an even size.
 *
 * Time Complexity: O(L log K), where L is the length of the string and K is the
 *                  number of unique characters (the alphabet size). Each map
 *                  insertion/access takes O(log K) time.
 * Space Complexity: O(K), to store the unique characters in the map.
 */
#include <iostream>
#include <map>
#include <string>

using namespace std;

int main() {

  string cad;
  // A map to store characters as keys and their frequencies as values.
  map<char, int> caracteres;
  cin >> cad;
  /**
   * @brief Populates the map with character frequencies.
   * The map's property of having unique keys is used to count distinct characters.
   */
  for (int i = 0; i < cad.size(); i++) {
    caracteres[cad[i]]++;
  }

  /**
   * @brief Checks the parity of the number of distinct characters.
   * `caracteres.size()` returns the number of unique elements in the map.
   * `& 1` is a bitwise check for oddness.
   */
  if (caracteres.size() & 1) {
    // Odd number of distinct characters -> Male
    cout << "IGNORE HIM!" << endl;
  } else {
    // Even number of distinct characters -> Female
    cout << "CHAT WITH HER!" << endl;
  }

  return 0;
}