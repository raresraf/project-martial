#include <iostream>
using namespace std;
int main() {
  int n = 0;
  string s;
  cin >> s;
  if (int(s[0]) >= 97 && int(s[0]) <= 122)
    s[0] = int(s[0]) - 32;
  cout << s;
  return 0;
}
