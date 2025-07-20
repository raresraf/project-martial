#include <iostream>
#include <string>
using namespace std;
int main() {
  string s1, s2;
  int flag = 0;
  cin >> s1 >> s2;
  for (int i = 0; i < s1.length(); i++) {
    if (s1[i] >= 65 && s1[i] <= 90)
      s1[i] = s1[i] + 32;
    if (s2[i] >= 65 && s2[i] <= 90)
      s2[i] = s2[i] + 32;
  }
  for (int i = 0; i < s1.length(); i++) {
    if (s1[i] < s2[i]) {
      flag = -1;
      break;
    } else if (s1[i] > s2[i]) {
      flag = 1;
      break;
    }
  }
  cout << flag;
  return 0;
}
