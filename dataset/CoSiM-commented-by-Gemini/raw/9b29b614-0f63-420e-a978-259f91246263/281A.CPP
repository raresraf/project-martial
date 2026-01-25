#include <iostream>

using namespace std;

int main() {

  string s = "";

  getline(cin, s);

  if (s[0] >= 97 && s[0] <= 122) {

    s[0] = s[0] - 32;
  }

  cout << s;

  return 0;
}
