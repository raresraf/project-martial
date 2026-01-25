// Written By Deepesh Nimma
// Codeforces: 118A Codeforces
#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define end1 "\n"

int main() {
  string s;
  cin >> s;

  transform(s.begin(), s.end(), s.begin(), ::tolower);

  s.erase(remove(s.begin(), s.end(), 'a'), s.end());
  s.erase(remove(s.begin(), s.end(), 'e'), s.end());
  s.erase(remove(s.begin(), s.end(), 'i'), s.end());
  s.erase(remove(s.begin(), s.end(), 'o'), s.end());
  s.erase(remove(s.begin(), s.end(), 'u'), s.end());
  s.erase(remove(s.begin(), s.end(), 'y'), s.end());

  for (int i = 0; i < s.size(); ++i) {
    cout << '.' << s[i];
  }
}
