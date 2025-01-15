#include <iostream>
#include <string>
#include <algorithm>
using namespace std;
int main() {
  string s;
  cin >> s;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '+')
      s.erase(i, 1);
  }
  sort(s.begin(), s.end());
  for (int i = 1; i < s.length(); i += 2)
    s.insert(i, 1, '+');
  cout << s;
  return 0;
}
