#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  string str;
  cin >> str;
  int r = 0, g = 0, b = 0;
  for (int i = 0; i < n; i++) {
    if (str[i - 1] == str[i]) {
      if (str[i] == 'R')
        ++r;
      else if (str[i] == 'G')
        ++g;
      else if (str[i] == 'B')
        ++b;
    }
  }
  cout << r + g + b;
}
