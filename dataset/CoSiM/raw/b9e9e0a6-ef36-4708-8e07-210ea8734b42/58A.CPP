#include <iostream>
#include <string>
using namespace std;
void solved() {
  string hello = "ello";
  string s;
  cin >> s;
  if (s.length() < 5) {
    cout << "NO";
    return;
  }
  for (int i = 0; i < s.length() - 4; i++) {
    if (s[i] == 'h' && s[i] != s[i + 1]) {
      int temp = 0;
      int check = false;
      for (int j = i + 1; j < s.length(); j++) {
        if (s[j] == s[j + 1] && s[j] == 'l' && check == false) {
          check = true;
          temp += 2;
          j++;
          continue;
        } else {
          if (s[j] == s[j + 1]) {
            continue;
          }
          if (s[j] == hello[temp]) {
            temp++;
          }
          if (temp == 4) {
            cout << "YES";
            return;
          }
        }
      }
    }
  }
  cout << "NO";
}
int main() {
  solved();
  return 0;
}
