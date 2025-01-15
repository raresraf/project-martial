#include <bits/stdc++.h>

using namespace std;

#define ll long long

int main() {
  string s;
  cin >> s;
  int c = s.length();
  int a = 0;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == 'h') {
      c = i;
      a++;
      break;
    }
  }
  for (int i = c + 1; i < s.length(); i++) {
    if (s[i] == 'e') {
      c = i;
      a++;
      break;
    }
  }
  for (int i = c + 1; i < s.length(); i++) {
    if (s[i] == 'l') {
      c = i;
      a++;
      break;
    }
  }
  for (int i = c + 1; i < s.length(); i++) {
    if (s[i] == 'l') {
      c = i;
      a++;
      break;
    }
  }
  for (int i = c + 1; i < s.length(); i++) {
    if (s[i] == 'o' && a == 4) {
      cout << "YES"
           << "\n";
      return 0;
    }
  }

  cout << "NO"
       << "\n";
  return 0;
}
