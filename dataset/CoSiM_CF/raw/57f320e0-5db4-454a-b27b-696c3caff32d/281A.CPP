#include <cstring>
#include <iostream>
using namespace std;
int main() {
  string s;
  cin >> s;
  if ((int)s[0] >= 97)
    s[0] = (int)s[0] - 32;
  cout << s;
  return 0;
}
