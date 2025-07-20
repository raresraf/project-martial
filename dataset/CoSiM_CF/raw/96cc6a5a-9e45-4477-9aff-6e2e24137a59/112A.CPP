#include <bits/stdc++.h>

using namespace std;

#define ll long long

char down(char s) { return (s <= 'Z' ? s + 32 : s); }

int main() {
  string s1, s2;
  cin >> s1 >> s2;
  for (int i = 0; i < s1.length(); i++) {
    s1[i] = down(s1[i]);
  }
  for (int i = 0; i < s1.length(); i++) {
    s2[i] = down(s2[i]);
  }
  for (int i = 0; i < s1.length(); i++) {
    if (s1[i] > s2[i]) {
      cout << 1 << "\n";
      return 0;
    } else if (s1[i] < s2[i]) {
      cout << -1 << "\n";
      return 0;
    }
  }
  cout << 0 << "\n";
  return 0;
}
