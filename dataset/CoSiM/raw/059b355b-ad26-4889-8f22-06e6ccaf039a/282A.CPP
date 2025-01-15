#include <iostream>
#define pf cout
#define sf cin
using namespace std;
int main() {
  int t;
  sf >> t;
  string s;
  int x = 0;
  while (t--) {

    sf >> s;

    if (s[1] == '-')
      x--;

    else
      x++;
  }
  pf << x;
}
