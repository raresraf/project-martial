// 339A
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  string s, r;
  cin >> s;
  int count1 = 0, count2 = 0, count3 = 0;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '1') {
      count1++;
    } else if (s[i] == '2') {
      count2++;
    } else if (s[i] == '3') {
      count3++;
    }
  }
  for (int i = 0; i < count1; i++) {
    r.push_back('1');
    r.push_back('+');
  }
  for (int i = 0; i < count2; i++) {
    r.push_back('2');
    r.push_back('+');
  }
  for (int i = 0; i < count3; i++) {
    r.push_back('3');
    r.push_back('+');
  }
  for (int i = 0; i < r.length() - 1; i++)
    cout << r[i];
  return 0;
}
