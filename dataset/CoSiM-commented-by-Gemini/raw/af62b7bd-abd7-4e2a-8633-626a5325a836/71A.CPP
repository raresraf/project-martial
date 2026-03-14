/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from Codeforces.
 * @see https://codeforces.com/problemset/problem/71/A
 *
 * This program processes a list of words. If a word is longer than 10 characters,
 * it is abbreviated by taking the first character, the number of characters
 * between the first and last, and the last character. Otherwise, the word is
 * printed as is.
 */
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <utility>
#include <vector>
#define MX 100000
#define fs first
#define sec second
#define TC                                                                     
  int T;                                                                       
  cin >> T;                                                                    
  while (T--)
#define loop(i, a, c) for (int i = 0; i < a; i++)
#define loop1(i, a, c) for (int i = 1; i <= a; i++)
#define pb(a) push_back(a)
#define all(a) (a).begin(), (a).end()
#define mp(x, y) make_pair((x), (y))
#define ll long long int
using namespace std;
int main() {
  int t;
  cin >> t;
  while (t--) {
    string s;
    cin >> s;
    /**
     * @brief If the word's length is 10 or less, print it as is.
     */
    if (s.length() <= 10) {
      cout << s << endl;
    } 
    /**
     * @brief If the word is longer than 10 characters, create an abbreviation.
     * The abbreviation is the first character, the number of middle characters,
     * and the last character.
     */
    else {
      cout << s[0];
      cout << s.length() - 2;
      cout << s[s.length() - 1] << endl;
    }
  }
}
