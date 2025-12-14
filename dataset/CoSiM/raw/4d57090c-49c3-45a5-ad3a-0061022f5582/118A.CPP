#include "bits/stdc++.h"
// PRAY :DOLPHINGARLIC: orz orz orz orz orz
// PRAY :DORI: orz orz orz orz orz orz orz
// PRAY :SAHIL KUCHLOUS: orz orz orz orz orz
// pray :stefan: orz orz orz orz orz orz orz
// pray :foshy: orz orz orz orz orz orz orz
#define GS                                                                     \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(nullptr);                                                            \
  cout.tie(0)
#define all(x) x.begin(), x.end()
#define el '\n'
#define Int long long
#define elif else if
#define test                                                                   \
  Int t;                                                                       \
  cin >> t;                                                                    \
  while (t--)
// #define

using namespace std;

int main() {
  GS;
  string s;
  cin >> s;
  vector<char> v;
  for (int i{}; i < (int)s.size(); i++) {
    s[i] = tolower(s[i]);
    if (s[i] != 'a' && s[i] != 'o' && s[i] != 'y' && s[i] != 'e' &&
        s[i] != 'u' && s[i] != 'i') {
      v.push_back('.');
      v.push_back(s[i]);
    }
  }
  for (auto i : v) {
    cout << i;
  }
  cout << endl;
  return 0;
}
/*
12:43 AM
9/18/2020
*/
