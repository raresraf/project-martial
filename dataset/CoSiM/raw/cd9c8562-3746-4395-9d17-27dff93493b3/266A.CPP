#include <bits/stdc++.h>

using namespace std;

#define ll long long

int main() {
  int n;
  cin >> n;
  string s;
  cin >> s;
  bool r = false;
  bool g = false;
  bool b = false;
  int ris = 0;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == 'R') {
      if (r)
        ris++;
      r = true;
      g = false;
      b = false;
    } else if (s[i] == 'G') {
      if (g)
        ris++;
      g = true;
      r = false;
      b = false;
    } else if (s[i] == 'B') {
      if (b)
        ris++;
      b = true;
      g = false;
      r = false;
    }
  }

  cout << ris << "\n";
  return 0;
}
