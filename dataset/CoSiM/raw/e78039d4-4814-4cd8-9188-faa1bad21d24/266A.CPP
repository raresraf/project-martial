#include <bits/stdc++.h>
#include <string>
using namespace std;

int main() {
  int n, cnt = 0;
  cin >> n;
  char s[n];
  cin >> s;
  for (int i = 0; i < n; i++)
    if (s[i] == s[i + 1])
      cnt++;
  cout << cnt;
  return 0;
}
