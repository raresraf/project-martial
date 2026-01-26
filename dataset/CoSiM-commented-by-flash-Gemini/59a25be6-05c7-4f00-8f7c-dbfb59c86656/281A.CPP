/*
 *	User: Isanchez_Aguilar
 *	Problem: CodeForces 281A - Word Capitalization
 */
#include <bits/stdc++.h>

using namespace std;

int main(void) {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);

  string s;
  cin >> s;

  s[0] = toupper(s[0]);

  cout << s << "\n";

  return 0;
}