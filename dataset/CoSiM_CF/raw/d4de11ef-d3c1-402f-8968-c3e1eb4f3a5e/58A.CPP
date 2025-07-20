#include <bits/stdc++.h>
using namespace std;

int main() {

  string s;
  cin >> s;
  int j = 0, tot = 0;
  for (int i = j; i < s.length(); i++) {
    if (s[i] == 'h') {
      j = i;
      tot++;
      break;
    }
  }
  for (int i = j + 1; i < s.length(); i++) {
    if (s[i] == 'e') {
      j = i;
      tot++;
      break;
    }
  }
  for (int i = j + 1; i < s.length(); i++) {
    if (s[i] == 'l') {
      j = i;
      tot++;
      break;
    }
  }
  for (int i = j + 1; i < s.length(); i++) {
    if (s[i] == 'l') {
      j = i;
      tot++;
      break;
    }
  }
  for (int i = j + 1; i < s.length(); i++) {
    if (s[i] == 'o') {
      j = i;
      tot++;
      break;
    }
  }
  if (tot == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
