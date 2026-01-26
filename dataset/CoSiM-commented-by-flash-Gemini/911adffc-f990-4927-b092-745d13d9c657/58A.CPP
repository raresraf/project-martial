// Codeforces
// 58A - Chat room

#include <iostream>
using namespace std;

int main() {
  string s;
  cin >> s;

  unsigned int i = 0, j = 0;

  string g = "hello";
  string output = "NO";

  while (i < s.size()) {
    if (s[i++] == g[j]) {
      j++;
    }

    if (j >= g.size()) {
      output = "YES";
      break;
    }
  }

  cout << output << endl;

  return 0;

  return 0;
}
