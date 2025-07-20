#include <bits/stdc++.h>

using namespace std;

#define ll long long

int main() {
  string s;
  cin >> s;
  int d = 0, r = 0;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '4')
      d++;
    else if (s[i] == '7')
      r++;
  }
  int q = d + r;
  if (q == 4 || q == 44 || q == 7 || q == 47 || q == 77 || q == 74 ||
      q == 444 || q == 447 || q == 474 || q == 477 || q == 744 || q == 747 ||
      q == 777) {
    cout << "YES"
         << "\n";
    return 0;
  }
  cout << "NO"
       << "\n";
  return 0;
}
