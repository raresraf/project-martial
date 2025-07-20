/*
 *	User: Isanchez_Aguilar
 *	Problem: CodeForces 71A - Way Too Long Words
 */
#include <bits/stdc++.h>

using namespace std;

int main(void) {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  int testCases;
  cin >> testCases;
  while (testCases--) {
    string word;
    cin >> word;
    int length = word.length();
    if (length > 10)
      cout << word[0] << length - 2 << word[length - 1] << "\n";
    else
      cout << word << "\n";
  }
  return 0;
}