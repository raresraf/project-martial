#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
using namespace std;
int main() {
  string s;
  cin >> s;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '+') {
      s.erase(s.begin() + i);
    }
  }
  sort(s.begin(), s.end());
  int len = s.length();
  int it = 1;
  for (int i = 0; i < len - 1; i++) {
    s.insert(it, "+");
    it += 2;
  }
  cout << s;
  return 0;
}
