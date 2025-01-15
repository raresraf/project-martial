#include <bits/stdc++.h>
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  string s1, s2;
  cin >> s1;
  cin >> s2;
  int flag = 0;

  for (int i = 0; i < s1.length(); ++i) {
    if (tolower(s1[i]) < tolower(s2[i])) {
      flag = -1;
      break;
    }
    if (tolower(s1[i]) > tolower(s2[i])) {
      flag = 1;
      break;
    }
  }
  cout << flag << "\n";

  return 0;
}