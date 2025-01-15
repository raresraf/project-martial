#include <iostream>

using namespace std;

int main() {
  string s, t;
  cin >> s >> t;
  for (int i = 0; i < s.size(); ++i) {
    if (s[i] >= 'A' && s[i] <= 'Z')
      s[i] = s[i] + 'a' - 'A';
    if (t[i] >= 'A' && t[i] <= 'Z')
      t[i] = t[i] + 'a' - 'A';
    if (s[i] > t[i]) {
      cout << 1;
      return 0;
    } else if (s[i] < t[i]) {
      cout << -1;
      return 0;
    }
  }
  cout << 0;
  return 0;
}