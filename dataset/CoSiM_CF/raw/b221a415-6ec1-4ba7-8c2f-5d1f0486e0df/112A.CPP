#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s1, s2;
  cin >> s1;
  cin >> s2;

  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);

  bool check = false;
  for (int i = 0; i < s1.length(); i++) {
    if (s1[i] < s2[i]) {
      cout << "-1";
      check = true;
      break;
    } else if (s1[i] == s2[i]) {
      continue;
    } else {
      cout << "1";
      check = true;
      break;
    }
  }

  if (!check) {
    cout << "0";
  }
  return 0;
}
