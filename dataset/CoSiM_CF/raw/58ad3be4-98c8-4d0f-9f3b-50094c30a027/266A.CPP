#include <bits/stdc++.h>
using namespace std;

int main() {
  int i, n, count = 0;
  char s[100];
  cin >> n;
  for (i = 0; i < n; i++)
    cin >> s[i];
  for (i = 0; i < n; i++) {
    if (s[i] == s[i + 1])
      count++;
  }
  cout << count;

  return 0;
}
