#include "string"
#include <iostream>

using namespace std;

int main(int argc, const char *argv[]) {
  string s;
  cin >> s;

  for (int i = 0; i < s.length(); i++) {
    for (int j = i + 2; j < s.length(); j += 2) {
      if (s[i] > s[j]) {
        int tmp = s[i];
        s[i] = s[j];
        s[j] = tmp;
      }
    }
  }
  cout << s;
  return 0;
}
