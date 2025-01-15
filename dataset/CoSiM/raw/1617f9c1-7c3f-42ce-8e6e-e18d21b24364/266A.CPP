// author @Nishant

#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, count = 0;
  string s;
  cin >> n >> s;
  char temp = s[0];
  for (int i = 1; i < n; i++) {
    if (temp == s[i]) {
      count++;
    }
    temp = s[i];
  }
  cout << count;
  return 0;
}