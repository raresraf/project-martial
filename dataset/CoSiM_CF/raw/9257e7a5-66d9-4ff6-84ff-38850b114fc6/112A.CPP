#include <iostream>
using namespace std;
int main() {
  string s, g;
  cin >> s >> g;
  int c = 10;
  for (int i = 0; i < s.length(); i++) {
    if (g[i] >= 65 && g[i] <= 90) {
      g[i] = g[i] - 'A' + 'a';
    }
    if (s[i] >= 65 && s[i] <= 90) {
      s[i] = s[i] - 'A' + 'a';
    }
    if (s[i] == g[i]) {
      c = 0;
      continue;
    }
    if (s[i] > g[i]) {
      c = 1;
      break;
    } else if (s[i] < g[i]) {
      c = -1;
      break;
    }
  }
  cout << c << endl;
}