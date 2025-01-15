#include <iostream>

using namespace std;

int main() {
  string s;
  cin >> s;
  int t;
  for (int i = 0; i < s.length(); i = i + 2) {
    for (int j = i; j < s.length(); j = j + 2) {
      if (s[i] > s[j]) {
        t = s[i];
        s[i] = s[j];
        s[j] = t;
      }
    }
  }
  cout << s;
}
