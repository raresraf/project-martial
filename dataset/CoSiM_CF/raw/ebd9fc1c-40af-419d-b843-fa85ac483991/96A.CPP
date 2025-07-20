#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  cin >> s;
  bool check = true;
  char curr = s[0];
  int count = 1;
  for (int i = 1; i < s.length(); i++) {
    if (s[i] == curr) {
      count++;
    } else {
      curr = s[i];
      count = 1;
    }
    if (count == 7) {
      check = false;
      break;
    }
  }
  if (check)
    cout << "NO";
  else
    cout << "YES";
}