#include <iostream>
#include <string>
using namespace std;

int main() {
  int n, i;
  int r = 0, b = 0, g = 0;
  string s;
  cin >> n;
  cin >> s;
  for (i = 0; i < n; i++) {
    if (s[i] == s[i + 1]) {
      if (s[i] == 'R')
        r++;
      else if (s[i] == 'B')
        b++;
      else if (s[i] == 'G')
        g++;
    }
  }
  cout << r + b + g;
  return 0;
}