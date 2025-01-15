#include <iostream>
#include <string.h>
//#include<ctype.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  string s;
  int x = 0;
  while (n--) {
    cin >> s;
    if (((s[0] == '+') && (s[1] == '+') && (s[2] == 'X')) ||
        ((s[0] == 'X') && (s[1] == '+') && (s[2] == '+'))) {
      x++;
    }
    //(s[0]==)
    else {
      x--;
    }
  }
  cout << x << endl;

  return 0;
}