#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  string s;
  cin >> n >> s;
  int r = 0;
  for (int i = 1; i < n; i++) {
    int j = i - 1;
    if (s[i] == s[j])
      r++;
  }
  cout << r;
}