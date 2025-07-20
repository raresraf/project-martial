#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main() {

  string s;
  int count = 1;
  cin >> s;
  for (int i = 1; i <= s.length(); i++) {
    if (s[i] == s[i - 1]) {
      count = count + 1;
      if (count == 7) {
        break;
      }
    } else {
      count = 1;
    }
  }

  if (count >= 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}