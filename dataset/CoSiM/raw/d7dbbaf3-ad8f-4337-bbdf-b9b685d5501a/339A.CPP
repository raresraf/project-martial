#include <algorithm>
#include <iostream>

using namespace std;

int main() {
  string s;
  cin >> s;
  sort(s.begin(), s.end());

  for (auto it = s.begin(); it != s.end(); ++it) {
    char c = (*it);
    if (c != '+') {
      cout << c;
      if ((it + 1) != s.end())
        cout << "+";
    }
  }
  return 0;
}
