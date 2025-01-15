#include <iostream>
#include <string>

using namespace std;

int main() {

  string s;
  cin >> s;
  int nOne = 0;
  int nTwo = 0;
  int nThree = 0;
  int i;
  for (i = 0; i < s.size(); i += 2) {

    if (s[i] == '1')
      nOne++;
    else if (s[i] == '2')
      nTwo++;
    else
      nThree++;
  }

  int size = s.size();
  string ans;
  while (nOne--) {
    ans += '1';
    if (size != 1)
      ans += '+';
    size--;
  }
  while (nTwo--) {
    ans += '2';
    if (size != 1)
      ans += '+';
    size--;
  }

  while (nThree--) {
    ans += '3';
    if (size != 1)
      ans += '+';
    size--;
  }

  cout << ans;

  return 0;
}