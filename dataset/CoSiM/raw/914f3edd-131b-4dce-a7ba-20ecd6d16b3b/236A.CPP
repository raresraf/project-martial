#include <iostream>
#include <string>
using namespace std;

int main() {
  string s;
  int i, c = 0, b[500] = {};
  cin >> s;
  for (i = 0; i < s.length(); i++) {
    if (!b[s[i]]) {
      c++;
      b[s[i]]++;
    }
  }
  if (c % 2) {
    cout << "IGNORE HIM!" << endl;
  } else {
    cout << "CHAT WITH HER!" << endl;
  }

  return 0;
}
