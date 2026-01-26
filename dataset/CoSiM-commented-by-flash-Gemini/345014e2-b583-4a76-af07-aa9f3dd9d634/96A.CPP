// 96A
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  string s;
  int flag = 0;
  cin >> s;
  int count = 1;
  for (int i = 1; i < s.length(); i++) {
    if (s[i - 1] == s[i]) {
      count++;
    } else {
      count = 1;
    }
    if (count == 7) {
      cout << "YES";
      flag = 1;
      break;
    }
  }
  if (flag == 0) {
    cout << "NO";
  }
  return 0;
}
