#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  string s;
  int count = 0;

  cin >> n >> s;

  for (int i = 1; i < n; i++) {
    if (s[i] == s[i - 1]) {
      count++;
    }
  }

  cout << count;

  return 0;
}
