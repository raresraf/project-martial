#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  string s;
  cin >> s;

  sort(s.begin(), s.end());
  s.erase(remove(s.begin(), s.end(), '+'), s.end());

  for (int i = 0; i < s.size(); i++) {
    if (i == s.size() - 1)
      cout << s[i];
    else
      cout << s[i] << "+";
  }

  return 0;
}
