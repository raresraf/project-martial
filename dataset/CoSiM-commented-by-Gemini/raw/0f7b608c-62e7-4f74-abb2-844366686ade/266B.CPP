#include <algorithm>
#include <iostream>
#include <string>

using std::cin;
using std::cout;
using std::endl;
using std::string;

int main() {
  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;

  if (s.length() != n)
    return -1;

  for (int i = 0; i != t; ++i) {
    for (int j = 0; j != n - 1; ++j) {
      if (s[j] == 'B' && s[j + 1] == 'G') {
        std::swap(s[j], s[j + 1]);
        if (j + 1 != s.length() - 1) {
          j++;
        }
      }
    }
  }
  cout << s;
  return 0;
}
