#include "string"
#include <iostream>
using namespace std;

int main(int argc, const char *argv[]) {
  string s;
  cin >> s;

  if (s[0] >= 97 && s[0] <= 122)
    s[0] -= 32;
  cout << s;

  return 0;
}
