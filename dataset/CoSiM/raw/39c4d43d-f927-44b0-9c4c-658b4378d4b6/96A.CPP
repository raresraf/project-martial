#include <iostream>

using namespace std;

int main() {
  string s;
  cin >> s;
  if (s.size() < 7) {
    cout << "NO";
    return 0;
  }
  for (int i = 6; i < s.size(); ++i) {
    bool b = true;
    for (int j = i - 6; j < i; ++j)
      if (s[j] != s[j + 1])
        b = false;
    if (b == true) {
      cout << "YES";
      return 0;
    }
  }
  cout << "NO";
}