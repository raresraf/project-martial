#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, cunt = 0;
  cin >> n;
  char str[n];
  cin >> str;
  for (int i = 0; i < n; i++) {
    if (str[i] == str[i + 1]) {
      cunt++;
    }
  }
  cout << cunt;

  return 0;
}
