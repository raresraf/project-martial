#include <iostream>
#include <math.h>
using namespace std;

int main() {

  char ch1, ch2;
  string s1, s2;
  int f = 0;

  getline(cin, s1);
  getline(cin, s2);

  for (int i = 0; i < s1.length(); ++i) {
    ch1 = s1[i];
    ch2 = s2[i];
    if ((int)s1[i] >= 97 && (int)s1[i] <= 122) {
      ch1 = (int)s1[i] - 32;
    }
    if ((int)s2[i] >= 97 && (int)s2[i] <= 122) {
      ch2 = (int)s2[i] - 32;
    }
    if ((int)ch1 < (int)ch2) {
      f = -1;
      break;
    } else if ((int)ch1 > (int)ch2) {
      f = 1;
      break;
    }
  }

  cout << f;

  return 0;
}
