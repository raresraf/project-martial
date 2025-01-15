#include <iostream>
#include <string.h>
using namespace std;
int main() {
  string s;
  int one = 0, two = 0, three = 0;
  cin >> s;
  for (int i = 0; i < s.length(); i += 2) {
    if (s[i] == '1')
      one++;
    if (s[i] == '2')
      two++;
    if (s[i] == '3')
      three++;
  }
  int index = 0;
  for (int i = 0; i < one; i++) {
    s[index] = '1';
    index += 2;
  }
  for (int i = 0; i < two; i++) {
    s[index] = '2';
    index += 2;
  }
  for (int i = 0; i < three; i++) {
    s[index] = '3';
    index += 2;
  }
  cout << s;
  return 0;
}
