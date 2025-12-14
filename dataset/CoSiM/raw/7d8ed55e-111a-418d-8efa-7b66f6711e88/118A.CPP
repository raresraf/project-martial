#include <bits/stdc++.h>
using namespace std;
#define ll long long int
bool vowel(char c) {
  switch (c) {
  case 'a':
  case 'e':
  case 'i':
  case 'o':
  case 'u':
  case 'y':
    return true;
  default:
    return false;
  }
}
int main() {
  string s;
  cin >> s;
  transform(s.begin(), s.end(), s.begin(), ::tolower);
  vector<char> m;
  for (int i = 0; i < s.length(); i++) {
    if (!vowel(s[i]))
      m.push_back(s[i]);
  }
  for (int i = 0; i < m.size(); i++) {
    cout << "." << m[i];
  }
}
