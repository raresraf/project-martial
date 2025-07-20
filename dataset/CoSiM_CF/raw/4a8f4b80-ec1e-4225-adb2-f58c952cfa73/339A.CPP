#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  cin >> s;

  if (s.size() == 1) {
    cout << s;
  } else {
    for (int i = 0; i < s.size(); i += 2) {

      for (int j = (i + 2); j < s.size(); j += 2) {
        if (s[i] > s[j]) {
          char temp = 0;
          temp = s[i];
          s[i] = s[j];
          s[j] = temp;
        }
      }
    }
    cout << s;
  }

  return 0;
}