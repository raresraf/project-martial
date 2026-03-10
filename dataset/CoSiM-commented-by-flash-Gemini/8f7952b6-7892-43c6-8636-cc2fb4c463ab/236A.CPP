/**
 * @file 236A.CPP
 * @brief This program solves the "Boy or Girl" problem (Codeforces 236A).
 *   It determines whether a given username (string) corresponds to a "masculine" or "feminine" persona
 *   based on the parity of the number of distinct characters it contains.
 *   If the number of distinct characters is even, it's "feminine" ("CHAT WITH HER!").
 *   If odd, it's "masculine" ("IGNORE HIM!").
 *   Problem description: https://codeforces.com/problemset/problem/236/A
 * * Algorithm: Counting distinct characters using a hash set (`std::set`).
 *   Each character of the input string is inserted into a `std::set`, which automatically handles uniqueness.
 *   The size of the set then gives the count of distinct characters.
 * Time Complexity: O(L log D), where L is the length of the string and D is the number of distinct characters.
 *   Since D is at most 26 (for English alphabet), log D is a small constant, making it effectively O(L).
 * Space Complexity: O(D) to store the distinct characters in the set. Since D <= 26, it's effectively O(1).
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, including <iostream> and <set>.
using namespace std;

// This comment describes a previous thought process or concept learned, not directly part of the working solution documentation.
// Learned the concept of Sets in c++ , updated Solution using sets -

/**
 * @brief Main function to determine the persona based on distinct characters.
 * Functional Utility: Reads a username, counts its distinct characters, and outputs a message
 *                     ("CHAT WITH HER!" or "IGNORE HIM!") based on the count's parity.
 */
int main() {
  string s;
  // Block Logic: Reads the input username string.
  cin >> s;
  // Functional Utility: Declares a `std::set` of characters. `std::set` automatically stores
  //                     only unique elements, making it ideal for counting distinct characters.
  set<char> un;
  /**
   * Block Logic: Iterates through each character in the input string 's'.
   * Invariant: After each iteration, 'un' contains all unique characters encountered so far.
   */
  for (char ch : s) {
    un.insert(ch); // Inserts each character into the set; duplicates are ignored.
  }
  /**
   * Block Logic: Checks the parity of the number of distinct characters.
   * If the number of distinct characters (size of the set) is even, it prints "CHAT WITH HER!".
   * Otherwise (if odd), it prints "IGNORE HIM!".
   */
  if (un.size() % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
}

/*
 * Block Logic: This commented-out section represents an alternative or previous approach
 *              to counting distinct characters. It manually iterates through the string
 *              and maintains a character array (`c`) to store unique characters found so far,
 *              using a `flag` to check for duplicates.
 * Time Complexity (for this commented-out section): O(L * D_avg), where L is string length
 *   and D_avg is average number of unique characters found, as it performs linear scan for uniqueness.
 * Space Complexity (for this commented-out section): O(D_max) where D_max is max possible unique characters.
 *
int main () {
string s;
cin>>s;
char c [101];
c[0] = '\0';
int currentsize=0;

for(char ch : s){
        bool flag = false;
        for(int i = 0 ; i <= currentsize;i++){
                if(ch ==c[i]){
                        flag = true;
                        break;
                }
        }

                if(flag == false) {
                        c[currentsize] = ch;
                        currentsize++;
                }




}
if(currentsize % 2 == 0) cout<<"CHAT WITH HER!"<<endl;
else cout<<"IGNORE HIM!"<<endl;
}
*/
