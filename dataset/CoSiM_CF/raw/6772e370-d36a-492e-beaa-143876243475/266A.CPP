#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, count = 0;
  cin >> n;

  string s;
  cin >> s;

  for (int i = 0; i < n; i++) {
    if (s[i] == s[i + 1])
      count++;
  }
  if (count == n)
    count = n - 1;

  cout << count;
}
