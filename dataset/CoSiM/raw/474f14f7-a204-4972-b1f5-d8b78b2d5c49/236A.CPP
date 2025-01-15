#include <iostream>

using namespace std;

int main() {
  string s;
  int c = 0;
  cin >> s;
  for (int j = 97; j <= 122; j++) {
    for (int i = 0; i < s.length(); i++) {
      if (s[i] == j) {
        c++;
        break;
      }
    }
  }
  if (c % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
  return 0;
}
